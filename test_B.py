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

from run_B import *
from test_A import *


if __name__ == '__main__':
    # threshold = [0.06, 0.20, 0.03333333333333334]
    threshold = [0.06, 0.2575757575757576, 0.03333333333333334]

    # 读取数据
    train = pd.read_csv('/work/data/train_base_data.csv')
    test = pd.read_csv('/work/data/test_data.csv')
    data = pd.concat([train[:5], test]).reset_index(drop=True)

    data_A, test_A = data.copy(), test.copy()
    pred_A = Test(data_A, test_A)

    # 数据预处理
    data, num_f, ff, cat_f = data_preprocessing(data, mode='test')
    with open('num_f_b.txt', 'r') as f:
        lines = f.readlines()  # 读取所有行到一个列表中
        num_f = eval(lines[0])
        ff = eval(lines[1])
        cat_f = eval(lines[2])
        length = eval(lines[3])

    
    # 特征工程
    data = feature_engineering_B(data, num_f, ff, cat_f, mode='test')

    # 减少内存使用
    data = reduce_mem_usage(data)
    
    # 分离训练测试
    test_ids = test['ID'].values
    test = data[data['ID'].isin(test_ids)].reset_index(drop=True)    # 仅包含 test_ids 中的 ID

    # 加载特征重要性
    with open('feat_imp_b.pkl', 'rb') as f:
        feat_imp_df = pickle.load(f)

    pred = [[] for _ in range(3)]
    for i in range(3):
        # 加载保存的模型
        booster = lgb.Booster(model_file='/work/model_{}_b.txt'.format(i))
        feat_imp_df_i = feat_imp_df[i]
        # print(feat_imp_df_i.sort_values(['imp'])[-50:])
        features = feat_imp_df_i.sort_values(['imp'])[-length[i]:]['feat'].to_list()
        # 打印这些特征的列
        # print(features)
        pred[i] = booster.predict(test[features], num_iteration=booster.best_iteration)

    pred[0] = pred[0]*0.99 + pred_A[0]*0.01
    pred[1] = pred[1]*(1/3) + pred_A[1]*(2/3)
    pred[2] = pred[2]*0.01 + pred_A[2]*0.99

    # for i in range(3):
    #     pred[i] = pred[i] * 0.5 + pred_A[i] * 0.5

    pred_np = np.array(pred)
    # # 分别计算一下三个通道的预测值的均值
    # A_mean = np.mean(pred_np[0])
    # B_mean = np.mean(pred_np[1])
    # C_mean = np.mean(pred_np[2])
    # print(A_mean, B_mean, C_mean)

    test['CHANNEL_A'] = np.where(pred_np[0] >= threshold[0], 1, 0)
    test['CHANNEL_B'] = np.where(pred_np[1] >= threshold[1], 1, 0)
    test['CHANNEL_C'] = np.where(pred_np[2] >= threshold[2], 1, 0)

    import csv
    with open('/work/output.csv', newline='', mode='w') as outputFile:
        fieldnames = ['ID', 'CHANNEL_A', 'CHANNEL_B', 'CHANNEL_C']
        writer = csv.DictWriter(outputFile, fieldnames=fieldnames)
        writer.writerow({'ID': 'ID', 'CHANNEL_A': 'CHANNEL_A', 'CHANNEL_B': 'CHANNEL_B', 'CHANNEL_C': 'CHANNEL_C'})
        for index, row in test[['ID', 'CHANNEL_A', 'CHANNEL_B', 'CHANNEL_C']].iterrows():
            writer.writerow({'ID': row['ID'], 'CHANNEL_A': row['CHANNEL_A'], 'CHANNEL_B': row['CHANNEL_B'], 'CHANNEL_C': row['CHANNEL_C']})
    print(test[['ID', 'CHANNEL_A']].head())