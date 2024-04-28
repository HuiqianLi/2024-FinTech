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

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, fbeta_score
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import LatentDirichletAllocation, NMF, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec

warnings.filterwarnings('ignore')

from run_A import *

# 特征工程
# 1. 偏离值特征
# 2. 数值和类别特征交叉
# 3. 加减乘除交叉
def feature_engineering_B(data, num_f, ff, cat_f, mode='train'):
    print('the num of ff num_f cat_f ', len(ff), len(num_f), len(cat_f))  # 33 0 1
    for group in tqdm(cat_f):
        for i, feature in enumerate(ff, start=1):  # 添加序号计数，从1开始
            if feature not in cat_f:
                tmp = data.groupby(group)[feature].agg(['mean', 'std', 'max', 'min', 'sum']).reset_index()
                tmp = pd.merge(data, tmp, on=group, how='left')
                # 创建新的特征，表示相对于组内统计的偏差
                data['{}-mean_gb_{}'.format(feature, group)] = data[feature] - tmp['mean']
                data['{}-min_gb_{}'.format(feature, group)] = data[feature] - tmp['min']
                data['{}-max_gb_{}'.format(feature, group)] = data[feature] - tmp['max']
                data['{}/sum_gb_{}'.format(feature, group)] = data[feature] / tmp['sum']   
    
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

if __name__ == '__main__':
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

    # 读取上次训练保存下来的特征列名
    with open('num_f.txt', 'r') as f:
        lines = f.readlines()  # 读取所有行到一个列表中
        num_f = eval(lines[0])
        ff = eval(lines[1])
        cat_f = eval(lines[2])
        length = eval(lines[3])

    # 读取特征重要性, 只处理这些特征
    with open('feat_imp_a.pkl', 'rb') as f:
        feat_imp_df = pickle.load(f)
    feature_col = []
    for i in range(3):
        feat_imp_df_i = feat_imp_df[i]
        features = feat_imp_df_i.sort_values(['imp'])[-length[i]:]['feat'].to_list()
        feature_col.extend(features)
    feature_col = list(set(feature_col))
    data = data[feature_col]
    num_f = [i for i in num_f if i in feature_col]
    ff = [i for i in ff if i in feature_col]
    cat_f = [i for i in cat_f if i in feature_col]

    # 数据预处理
    data, num_f, ff, cat_f = data_preprocessing(data)
    # 把num_f, ff, cat_f保存在文件中
    with open('num_f_b.txt', 'w') as f:
        f.write(str(num_f) + '\n')
        f.write(str(ff) + '\n')
        f.write(str(cat_f) + '\n')

    # 特征工程
    data = feature_engineering_B(data, num_f, ff, cat_f)

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


    # # 临时训练一下
    # # 加载特征重要性
    # with open('feat_imp_b.pkl', 'rb') as f:
    #     feat_imp_df = pickle.load(f)
    # # 筛选大于1的特征
    # imp = [2, 50, 30]
    # for i in range(3):
    #     feat_imp_df_i = feat_imp_df[i]
    #     # 按照特征重要性排序
    #     # print(feat_imp_df_i.sort_values(['imp'])[-50:])
    #     features_ = feat_imp_df_i[feat_imp_df_i['imp'] > imp[i]]['feat'].to_list()
    #     length[i] = len(features_)
    # print('============here is the length of features ', length)
    # pred = [[] for _ in range(3)]
    # oof = [[] for _ in range(3)]
    # for i in range(3):
    #     print('[Channel {}]'.format(i))
    #     feat_imp_df_i = feat_imp_df[i]
    #     pred[i], oof[i] = mean_fusion(train, test, feat_imp_df_i, y.iloc[i], threshold[i], length[i], params[i], model_path='model_{}_b.txt'.format(i))


    # 模型全特征训练
    pred = [[] for _ in range(3)]
    oof = [[] for _ in range(3)]
    feat_imp_df = [None for _ in range(3)]
    for i in range(3):
        feat_imp_df[i], pred[i], oof[i] = full_feature(train, test, features, y.iloc[i], threshold[i], pred[i], oof[i], params[i])

    # 使用 pickle 保存列表到文件
    with open('feat_imp_b.pkl', 'wb') as f:
        pickle.dump(feat_imp_df, f)

    length = [0 for _ in range(3)]
    # 筛选大于1的特征
    imp = [2, 50, 30]
    for i in range(3):
        feat_imp_df_i = feat_imp_df[i]
        features_ = feat_imp_df_i[feat_imp_df_i['imp'] >= imp[i]]['feat'].to_list()
        length[i] = len(features_)
    print('============here is the length of features ', length)
    # 把length也写进num_f.txt文件中
    with open('num_f_b.txt', 'a') as f:
        f.write(str(length) + '\n')


    pred = [[] for _ in range(3)]
    oof = [[] for _ in range(3)]
    for i in range(3):
        print('[Channel {}]'.format(i))
        feat_imp_df_i = feat_imp_df[i]
        pred[i], oof[i] = mean_fusion(train, test, feat_imp_df_i, y.iloc[i], threshold[i], length[i], params[i], model_path='model_{}_b.txt'.format(i))
