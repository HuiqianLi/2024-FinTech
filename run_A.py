import re
import os
import sklearn
import json
import pandas as pd
import warnings
import multiprocessing
import toad

import lightgbm as lgb
import numpy as np
from tqdm import tqdm
import pickle

from collections import defaultdict

from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, fbeta_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import LatentDirichletAllocation, NMF, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer

warnings.filterwarnings('ignore')

from config import params

# 数据预处理
# 编码、缺失值等
# 数值型
def data_preprocessing(data, mode='train'):
    ff = [i for i in data.columns if i not in ['ID', 'CHANNEL_A', 'CHANNEL_B', 'CHANNEL_C']]
    # 需要特别处理的类别特征的列名
    cat_f = ['COL3', 'COL4', 'COL5', 'COL19']
    num_f = []  # 存储数值型特征的列名
    for f in tqdm(ff):
        data[f] = data[f].fillna(-2)
        data[f] = data[f].astype('str')
        data[f] = data[f].apply(lambda x: x.replace(' ', '-1'))
        if f not in cat_f:
            data[f] = data[f].astype('float')
        else:
            data[f] = data[f].astype('str')
            # 对类别特征进行Label Encoding
            if data[f].dtype == 'object':
                lb = LabelEncoder()
                data[f] = lb.fit_transform(data[f])
            else:
                grade_dict = {chr(i): i-96 for i in range(97, 123)}
                data[f] = data[f].map(grade_dict)
        if data[f].max()>1000 and mode != 'test':
            num_f.append(f)

    print('after preprocessing, the num of ff num_f cat_f ', len(ff), len(num_f), len(cat_f))  # 54 50 4

    # 去掉低方差、高方差特征
    if mode != 'test':
        data, num_f, ff, cat_f = remove_features(data, num_f, ff, cat_f)
    
    # # 数据标准化，归一化到0，1之间
    # for f in ff:
    #     data[f] = (data[f] - data[f].min()) / (data[f].max() - data[f].min())

    return data, num_f, ff, cat_f


# 去掉低方差、高方差特征
def remove_features(data, num_f, ff, cat_f):
    # 去除低方差特征
    variance_threshold = VarianceThreshold(threshold=0.6)  # 设置阈值为1
    selected_columns = variance_threshold.fit_transform(data[ff])  # 删除方差小于等于threshold的特征
    selected_ff = variance_threshold.get_support(indices=True)  # 获取被保留下来的列的索引
    selected_columns_df = pd.DataFrame(selected_columns, columns=[ff[i] for i in selected_ff])
    data = pd.concat([data.drop(columns=ff), selected_columns_df], axis=1)
    ff = list(selected_columns_df.columns)
    num_f = [col for col in num_f if col in ff]  # 更新数值型特征列表
    cat_f = [col for col in cat_f if col in ff]  # 更新cat_f
    print('after removing low variance, the num of ff num_f cat_f ', len(ff), len(num_f), len(cat_f))

    # 去除高方差特征（如果需要）
    # 这里需要指定一个高于平均方差的阈值，例如：
    high_variance_threshold = 3 * variance_threshold.variances_.mean()  # 假设3倍的平均方差作为高方差的阈值
    high_variance_indices = variance_threshold.variances_ > high_variance_threshold
    selected_high_variance_ff = [ff[i] for i in range(len(ff)) if high_variance_indices[i]]
    data = data.drop(columns=selected_high_variance_ff)
    ff = [col for col in ff if col not in selected_high_variance_ff]
    num_f = [col for col in num_f if col not in selected_high_variance_ff]
    cat_f = [col for col in cat_f if col not in selected_high_variance_ff]
    print('after removing high variance, the num of ff num_f cat_f ', len(ff), len(num_f), len(cat_f))

    return data, num_f, ff, cat_f


# 减少内存使用
def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in tqdm(df):
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df


# 特征工程
# 1. 偏离值特征
# 2. 数值和类别特征交叉
# 3. 加减乘除交叉
def feature_engineering(data, num_f, ff, cat_f, mode='train'):
    print('the num of ff num_f cat_f ', len(ff), len(num_f), len(cat_f))  # 33 0 1
    # for group in tqdm(cat_f):
    #     for i, feature in enumerate(ff, start=1):  # 添加序号计数，从1开始
    #         if feature not in cat_f:
    #             tmp = data.groupby(group)[feature].agg(['mean', 'std', 'max', 'min', 'sum']).reset_index()
    #             tmp = pd.merge(data, tmp, on=group, how='left')
    #             # 创建新的特征，表示相对于组内统计的偏差
    #             data['{}-mean_gb_{}'.format(feature, group)] = data[feature] - tmp['mean']
    #             data['{}-min_gb_{}'.format(feature, group)] = data[feature] - tmp['min']
    #             data['{}-max_gb_{}'.format(feature, group)] = data[feature] - tmp['max']
    #             data['{}/sum_gb_{}'.format(feature, group)] = data[feature] / tmp['sum']   
    
    # 数值型特征和类别特征之间的交叉
    for i in tqdm(range(len(num_f))):
        for j in range(i + 1, len(num_f)):
            for cat in cat_f[1:]:
                f1 = ff[i]
                f2 = ff[j]
                data[f'{f1}_{f2}_log_{cat}'] = (np.log1p(data[f1]) - np.log1p(data[f2])) * data[cat]
                data[f'{f1}+{f2}_log_{cat}'] = (np.log1p(data[f1]) + np.log1p(data[f2])) * data[cat]
                data[f'{f1}*{f2}_log_{cat}'] = (np.log1p(data[f1]) * np.log1p(data[f2])) * data[cat]
                data[f'{f1}/{f2}_log_{cat}'] = (np.log1p(data[f1]) / np.log1p(data[f2])) * data[cat]
                data[f'{f2}/{f1}_log_{cat}'] = (np.log1p(data[f2]) / np.log1p(data[f1])) * data[cat]

                data[f'{f1}_{f2}_log_{cat}_'] = (np.log1p(data[f1]) - np.log1p(data[f2])) / data[cat]
                data[f'{f1}+{f2}_log_{cat}_'] = (np.log1p(data[f1]) + np.log1p(data[f2])) / data[cat]
                data[f'{f1}*{f2}_log_{cat}_'] = (np.log1p(data[f1]) * np.log1p(data[f2])) / data[cat]
                data[f'{f1}/{f2}_log_{cat}_'] = (np.log1p(data[f1]) / np.log1p(data[f2])) / data[cat]
                data[f'{f2}/{f1}_log_{cat}_'] = (np.log1p(data[f2]) / np.log1p(data[f1])) / data[cat]

    # # 数值型特征之间的加减乘除交叉
    # for i in tqdm(range(len(num_f))):
    #     for j in range(i + 1, len(num_f)):
    #         f1 = ff[i]
    #         f2 = ff[j]
    #         data[f'{f1}_{f2}'] = data[f1] - data[f2]
    #         data[f'{f1}+{f2}'] = data[f1] + data[f2]
    #         data[f'{f1}*{f2}'] = data[f1] * data[f2]
    #         data[f'{f1}/{f2}'] = data[f1] / data[f2]
    #         data[f'{f2}/{f1}'] = data[f2] / data[f1]

    # 数值特征做 max, min, mean, std
    for i in tqdm(range(len(num_f))):
        f = ff[i]
        data[f'{f}_max'] = data[f].max()
        data[f'{f}_min'] = data[f].min()
        mean_series = data[f].mean()
        std_series = data[f].std()
        ptp_series = data[f].max() - data[f].min()  # 计算峰峰值
        data[f'{f}_mean'] = mean_series
        data[f'{f}_std'] = std_series
        data[f'{f}_ptp'] = ptp_series

    return data


# 训练模型
def train_model(X_train, X_test, features, y, threshold_, params, seed=2024, save_model=False, model_path='model.txt', kf=10):
    feat_imp_df = pd.DataFrame({'feat': features, 'imp': 0})    # 存储特征名称和它们的重要性
    KF = StratifiedKFold(n_splits=kf, random_state=seed, shuffle=True)   # 5折交叉验证
    # 初始化保存每个折的分数列表
    score_lists = []
    
    oof_lgb = np.zeros(len(X_train))    # 初始化1个任务的oof预测结果
    predictions_lgb = np.zeros(len(X_test))  # 测试集的预测结果，1个任务

    for fold_, (trn_idx, val_idx) in enumerate(KF.split(X_train.values, y.values)):
        print("[fold n°{}]".format(fold_ + 1))
        trn_data = lgb.Dataset(X_train.iloc[trn_idx][features], label=y.iloc[trn_idx])
        val_data = lgb.Dataset(X_train.iloc[val_idx][features], label=y.iloc[val_idx])

        clf = lgb.train(params, 
                        trn_data, 
                        valid_sets=[trn_data, val_data], 
                        verbose_eval=100)
        
        # model_lgb = lgb.LGBMClassifier(objective='binary', max_depth=3, num_leaves=50,
        #                     n_estimators=5000,
        #                     min_child_samples=18, min_child_weight=0.001,
        #                     feature_fraction=0.6, bagging_fraction=0.5,
        #                     metric='auc', )
        # params_test={
        #         'learning_rate=': [0.5, 0.1, 0.05, 0.01],
        #     }
        # gsearch = GridSearchCV(estimator=model_lgb, param_grid=params_test, scoring='f1_micro', cv=5, verbose=1, n_jobs=4)
        # gsearch.fit(X_train.iloc[trn_idx][features], y.iloc[trn_idx])
        # print(gsearch.best_params_, gsearch.best_score_)

        oof_lgb[val_idx] = clf.predict(X_train.iloc[val_idx][features], num_iteration=clf.best_iteration)
        predictions_lgb += clf.predict(X_test[features], num_iteration=clf.best_iteration) / KF.n_splits
        feat_imp_df['imp'] += clf.feature_importance() / KF.n_splits
        score_lists.append(f1_score(y.iloc[val_idx], [1 if i >= threshold_ else 0 for i in oof_lgb[val_idx]]))


    # 打印每个任务的评估指标
    print("AUC score: {}".format(roc_auc_score(y, oof_lgb)))
    print("F1 score: {}".format(f1_score(y, [1 if i >= threshold_ else 0 for i in oof_lgb])))
    print("Precision score: {}".format(precision_score(y, [1 if i >= threshold_ else 0 for i in oof_lgb])))
    print("Recall score: {}".format(recall_score(y, [1 if i >= threshold_ else 0 for i in oof_lgb])))
    print("F1 mean: {}".format(np.mean(score_lists)))

    # # 假设oof_lgb是模型输出的概率，y是真实标签
    # thresholds = np.linspace(0, threshold_+0.1, 100)  # 生成一系列可能的阈值
    # best_threshold = 0
    # best_f1 = 0

    # for threshold in thresholds:
    #     y_pred = [1 if i >= threshold else 0 for i in oof_lgb]
    #     current_f1 = f1_score(y, y_pred)
    #     if current_f1 > best_f1:
    #         best_f1 = current_f1
    #         best_threshold = threshold

    # print("Best F1 score: {}".format(best_f1))
    # print("Best threshold: {}".format(best_threshold))

    # # 使用最佳阈值计算其他指标
    # y_pred_best = [1 if i >= best_threshold else 0 for i in oof_lgb]
    # print("AUC score: {}".format(roc_auc_score(y, oof_lgb)))
    # print("F1 score with best threshold: {}".format(f1_score(y, y_pred_best)))
    # print("Precision score with best threshold: {}".format(precision_score(y, y_pred_best)))
    # print("Recall score with best threshold: {}".format(recall_score(y, y_pred_best)))

    if save_model:
        booster = lgb.train(params, trn_data, valid_sets=[trn_data, val_data], verbose_eval=100)
        booster.save_model(model_path)  # 保存模型到文件
    
    # 返回特征重要性、每个任务的oof预测结果和测试集的预测结果
    return feat_imp_df, oof_lgb, predictions_lgb


# 模型全特征训练, 得到特征重要性排名
def full_feature(train, test, features, y, threshold_, pred, oof, params):
    seeds = [2024]
    for seed in seeds:
        feat_imp_df, oof_lgb, predictions_lgb = train_model(train, test, features, y, threshold_, params, seed=seed, save_model=False, kf=2)
        pred.append(predictions_lgb)
        oof.append(oof_lgb)
    return feat_imp_df, pred, oof


# 重要性筛选
def mean_fusion(train, test, feat_imp_df, y, threshold_, length, params, model_path='model.txt'):
    features2 = feat_imp_df.sort_values(['imp'])[-length:]['feat'].to_list()
    _, oof_lgb, predictions_lgb = train_model(train, test, features2, y, threshold_, params, seed=2024, save_model=True, model_path=model_path)
    pred = predictions_lgb
    oof = oof_lgb
    return pred, oof


if __name__ == '__main__':
    # threshold = [0.06, 0.20, 0.03]
    threshold = [0.06, 0.2575757575757576, 0.03333333333333334]
    # 读取数据
    train = pd.read_csv('/work/data/train_base_data.csv')
    test = pd.read_csv('/work/data/test_data.csv')
    data = pd.concat([train, test]).reset_index(drop=True)

    # 暂时只选择data的前500行
    # data = pd.concat([train[:500], test]).reset_index(drop=True)

    y_label = ['ID', 'CHANNEL_A', 'CHANNEL_B', 'CHANNEL_C']
    y_data = data[y_label]
    data = data.drop(columns=y_label)

    # 数据预处理
    data, num_f, ff, cat_f = data_preprocessing(data)
    # 把num_f, ff, cat_f保存在文件中
    with open('num_f.txt', 'w') as f:
        f.write(str(num_f) + '\n')
        f.write(str(ff) + '\n')
        f.write(str(cat_f) + '\n')

    # 特征工程
    data = feature_engineering(data, num_f, ff, cat_f)

    # 减少内存使用
    data = reduce_mem_usage(data)
    
    # 分离训练测试
    data = pd.concat([data, y_data], axis=1)    # 将标签数据加回来
    test_ids = test['ID'].values
    train = data[~data['ID'].isin(test_ids)].reset_index(drop=True)  # 排除 test_ids 中的 ID
    test = data[data['ID'].isin(test_ids)].reset_index(drop=True)    # 仅包含 test_ids 中的 ID
    features = [i for i in data.columns if i not in ['ID', 'CHANNEL_A', 'CHANNEL_B', 'CHANNEL_C']]
    y = train[['CHANNEL_A', 'CHANNEL_B', 'CHANNEL_C']]    # (500000, 3)
    y = np.transpose(y)     # 转置y
    print("Train files: ", len(train), "| Test files: ", len(test), "| Feature nums", len(features))


    # 模型全特征训练
    pred = [[] for _ in range(3)]
    oof = [[] for _ in range(3)]
    feat_imp_df = [None for _ in range(3)]
    for i in range(3):
        feat_imp_df[i], pred[i], oof[i] = full_feature(train, test, features, y.iloc[i], threshold[i], pred[i], oof[i], params[i])

    # 保存特征重要性、oof预测结果
    # np.save('score_dict_a.npy', oof)
    # 使用 pickle 保存列表到文件
    with open('feat_imp_a.pkl', 'wb') as f:
        pickle.dump(feat_imp_df, f)

    length = [0 for _ in range(3)]
    # 筛选大于0.05的特征
    for i in range(3):
        feat_imp_df_i = feat_imp_df[i]
        features_ = feat_imp_df_i[feat_imp_df_i['imp'] > 0.05]['feat'].to_list()
        length[i] = len(features_)
    print('============here is the length of features ', length)
    # 把length也写进num_f.txt文件中
    with open('num_f.txt', 'a') as f:
        f.write(str(length) + '\n')


    # 保存特征
    # score_dict = np.load('score_dict_a.npy', allow_pickle=True)
    # with open('feat_imp_a.pkl', 'rb') as f:
    #     feat_imp_df = pickle.load(f)
    # feat_imp_df = pd.read_pickle('feat_imp_a.pkl')

    pred = [[] for _ in range(3)]
    oof = [[] for _ in range(3)]
    for i in range(3):
        print('[Channel {}]'.format(i))
        feat_imp_df_i = feat_imp_df[i]
        pred[i], oof[i] = mean_fusion(train, test, feat_imp_df_i, y.iloc[i], threshold[i], length[i], params[i], model_path='model_{}.txt'.format(i))

    # # 保存结果
    # pred_np = np.array(pred)
    # pred_binary = np.where(pred_np >= threshold, 1, 0)
    # test['CHANNEL_A'] = pred_binary[0]
    # test['CHANNEL_B'] = pred_binary[1]
    # test['CHANNEL_C'] = pred_binary[2]

    # import csv
    # with open('/work/output.csv', newline='', mode='w') as outputFile:
    #     fieldnames = ['ID', 'CHANNEL_A', 'CHANNEL_B', 'CHANNEL_C']
    #     writer = csv.DictWriter(outputFile, fieldnames=fieldnames)
    #     writer.writerow({'ID': 'ID', 'CHANNEL_A': 'CHANNEL_A', 'CHANNEL_B': 'CHANNEL_B', 'CHANNEL_C': 'CHANNEL_C'})
    #     for index, row in test[['ID', 'CHANNEL_A', 'CHANNEL_B', 'CHANNEL_C']].iterrows():
    #         writer.writerow({'ID': row['ID'], 'CHANNEL_A': row['CHANNEL_A'], 'CHANNEL_B': row['CHANNEL_B'], 'CHANNEL_C': row['CHANNEL_C']})
    # print(test[['ID', 'CHANNEL_A']].head())