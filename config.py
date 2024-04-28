# 设置LightGBM参数
seed = 2024
params = [{
    'objective': 'binary',
    'boosting_type': 'gbdt',
    'metric': {'binary_logloss', 'auc'},
    'n_jobs': 4,
    'learning_rate': 0.5,  # 0.1
    'num_leaves': 50,
    'max_depth': 5,
    'tree_learner': 'serial',
    'min_data_in_leaf': 18,
    'min_sum_hessian_in_leaf': 1e-2,
    'bagging_fraction': 0.6,
    'feature_fraction': 0.5,
    'num_boost_round': 5000,
    'max_bin': 255,
    'verbose': -1,
    'seed': seed,
    'bagging_seed': seed,
    'feature_fraction_seed': seed,
    'early_stopping_rounds': 100,
},
{
    'objective': 'binary',
    'boosting_type': 'gbdt',
    'metric': {'binary_logloss', 'auc'},
    'n_jobs': 4,
    'learning_rate': 0.1,  # 0.1
    'num_leaves': 2 ** 6,
    'max_depth': 8,
    'tree_learner': 'serial',
    'colsample_bytree': 0.8,
    'subsample_freq': 1,
    'subsample': 0.8,
    'num_boost_round': 5000,
    'max_bin': 255,
    'verbose': -1,
    'seed': seed,
    'bagging_seed': seed,
    'feature_fraction_seed': seed,
    'early_stopping_rounds': 100,
},
{
    'objective': 'binary',
    'boosting_type': 'gbdt', # 'gbdt'
    'metric': {'binary', 'auc'},
    'n_jobs': 4,
    'learning_rate': 0.05,  # 0.1
    'num_leaves': 2 ** 6, 
    'max_depth': 25, # 8
    'tree_learner': 'serial',
    'min_data_in_leaf ': 20, # 20
    'lambda_l1': 0.01,  # 正则化
    'lambda_l2': 10,  # 正则化
    'colsample_bytree': 0.8,
    'subsample_freq': 1,
    'subsample': 0.8,
    'num_boost_round': 5000,
    'max_bin': 255, # 用于连续特征分桶的最大桶数, 255
    'verbose': -1,
    'seed': seed,
    'bagging_seed': seed,
    'feature_fraction_seed': seed,
    'early_stopping_rounds': 200,
}]