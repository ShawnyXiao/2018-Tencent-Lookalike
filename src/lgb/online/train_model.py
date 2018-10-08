
# coding: utf-8

# In[1]:


from __future__ import division
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from scipy import sparse
import xgboost as xgb
import lightgbm as lgb
import cPickle
import time
import datetime
import math
import os
from multiprocessing import cpu_count
import gc
import warnings
warnings.filterwarnings('ignore')


# In[2]:


# Constants define
ROOT_PATH = '/home/xiaoxy/2018-Tencent-Lookalike/'
ONLINE = 1


# In[3]:


target = 'label'
train_len = 45539700  # 8798814
test1_len = 11729073  # 2265989
test2_len = 11727304  # 2265879
positive_num = 2182403  # 421961


# In[ ]:


########################################### Helper function ###########################################


# In[4]:


def log(info):
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' ' + str(info))


# In[5]:


def write_file_jq(df, path):
    SIZE = 1048576000
    size = df.memory_usage().sum()
    parts = size // SIZE + 1
    length = len(df)
    part = length // parts
    i = 0
    while (True):
        part_path = path + '_part{}.pkl.gz'.format(i)
        if os.path.exists(part_path):
            os.remove(part_path)
            i = i + 1
        else:
            break
    for i in range(parts):
        part_path = path + '_part{}.pkl.gz'.format(i)
        if os.path.exists(part_path):
            raise Exception('File Exists')
        cPickle.dump(df.iloc[part * i:part * (i + 1)], open(part_path, 'wb'))
    last = df.iloc[part * parts:]
    if len(last) > 0:
        cPickle.dump(last, open(path + '_part{}.pkl.gz'.format(parts), 'wb'))

def read_file_jq(path):
    i = 0
    arr = []
    while (True):
        part_path = path + '_part{}.pkl.gz'.format(i)
        if os.path.exists(part_path):
            arr.append(cPickle.load(open(open(part_path, 'rb'))))
            i = i + 1
        else:
            break
    if len(arr) == 0:
        raise Exception("No File")
    return pd.concat(arr)


# In[6]:


def lgb_cv(train_x, train_y, test_x, test_y, params, folds, rounds):
    start = time.clock()
    log('LightGBM run cv: ' + str(rounds) + ' rounds')
    # params['scale_pos_weight'] = float(len(train_y[train_y == 0])) / len(train_y[train_y == 1])
    dtrain = lgb.Dataset(train_x, label=train_y)
    dtest = lgb.Dataset(test_x, label=test_y)
    model = lgb.train(params, dtrain, rounds, valid_sets=[dtest], valid_names=['test'], verbose_eval=1, early_stopping_rounds=10)
    elapsed = (time.clock() - start)
    log('Time used: '+ str(elapsed) + ' s')
    return model, model.best_iteration, model.best_score['test']['auc']

def lgb_train_predict(train_x, train_y, test_x, params, rounds):
    start = time.clock()
    log('LightGBM train: ' + str(rounds) + ' rounds')
    dtrain = lgb.Dataset(train_x, label=train_y)
    model = lgb.train(params, dtrain, rounds, valid_sets=[dtrain], verbose_eval=1)
    elapsed = (time.clock() - start)
    log('Time used: '+ str(elapsed) + ' s')
    pred = model.predict(test_x)
    return model, pred

def store_result(result, pred, name):
    result['score'] = pred
    result['score'] = result['score'].apply(lambda x: float('%.6f' % x))
    result.to_csv(ROOT_PATH + 'data/output/lgb/final/sub/' + name + '.csv', index=False, sep=',')
    return result


# In[ ]:


########################################### Read data ###########################################


# In[13]:


df_basic = cPickle.load(open(ROOT_PATH + 'data/output/lgb/final/feat/train+test1+test2/all(basic).p', 'rb'))
df_stat1 = cPickle.load(open(ROOT_PATH + 'data/output/lgb/final/feat/train+test1+test2/all(stat1).p', 'rb'))
df_stat2 = cPickle.load(open(ROOT_PATH + 'data/output/lgb/final/feat/train+test1+test2/all(stat2).p', 'rb'))
df_stat3 = cPickle.load(open(ROOT_PATH + 'data/output/lgb/final/feat/train+test1+test2/all(stat3).p', 'rb'))
df_stat4 = cPickle.load(open(ROOT_PATH + 'data/output/lgb/final/feat/train+test1+test2/all(stat4).p', 'rb'))
df_stat = pd.concat([df_stat1, df_stat2, df_stat3, df_stat4], axis=1); del df_stat1, df_stat2, df_stat3, df_stat4; gc.collect()
df_aduser1 = cPickle.load(open(ROOT_PATH + 'data/output/lgb/final/feat/train+test1+test2/all(aduser1).p', 'rb'))
df_aduser2 = cPickle.load(open(ROOT_PATH + 'data/output/lgb/final/feat/train+test1+test2/all(aduser2).p', 'rb'))
df_aduser = pd.concat([df_aduser1, df_aduser2], axis=1); del df_aduser1, df_aduser2; gc.collect()
df_userad1 = cPickle.load(open(ROOT_PATH + 'data/output/lgb/final/feat/train+test1+test2/all(userad1).p', 'rb'))
df_userad2 = cPickle.load(open(ROOT_PATH + 'data/output/lgb/final/feat/train+test1+test2/all(userad2).p', 'rb'))
df_userad = pd.concat([df_userad1, df_userad2], axis=1); del df_userad1, df_userad2; gc.collect()
predictors_stat = list(df_stat.columns.values)
predictors_aduser = list(df_aduser.columns.values)
predictors_userad = list(df_userad.columns.values)
df = pd.concat([df_basic, df_stat, df_aduser, df_userad], axis=1)
del df_basic, df_stat, df_aduser, df_userad; gc.collect()

# Split dataset for online
train = df.loc[:(train_len-1), :]
test = df.loc[train_len:, :]
del df; gc.collect()

predictors = []
predictors.append('creativeSize')
predictors += predictors_stat + predictors_aduser + predictors_userad
train_x = train[predictors]
test_x = test[predictors]
train_y = train[target]
test_y = test[target]

train_cv = sparse.load_npz('%sdata/output/lgb/final/feat/train+test1+test2/train(cv)(online=%s).npz' % (ROOT_PATH, str(ONLINE)))
test_cv = sparse.load_npz('%sdata//output/lgb/final/feat/train+test1+test2/test(cv)(online=%s).npz' % (ROOT_PATH, str(ONLINE)))
train_x = sparse.hstack((train_x, train_cv)); del train_cv; gc.collect()
test_x = sparse.hstack((test_x, test_cv)); del test_cv; gc.collect()


# In[ ]:


########################################### LigthGBM ###########################################


# In[7]:


config_lgb = {
    'rounds': 10000,
    'folds': 5
}

params_lgb = {
    'boosting_type': 'gbdt',
    'objective': 'binary',  # 'xentropy'
    'metric': {'auc'},
    'num_leaves': 1023,
    'learning_rate': 0.01,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'min_sum_hessian_in_leaf': 10,
    'verbosity': 1,
    'num_threads': cpu_count() - 1,
    'seed': 7,
    # 'device': 'gpu'
}


# In[15]:


model_lgb, pred_lgb = lgb_train_predict(train_x, train_y, test_x, params_lgb, 1500)
model_lgb.save_model(ROOT_PATH + 'data/output/lgb/final/model/lgb-%d(r%d).txt' % (7403, 1500))
store_result(pd.read_csv(ROOT_PATH + 'data/input/final/test1.csv', header=0, sep=','), pred_lgb[:test1_len], '20180530-lgb-%d(r%d)' % (7403, 1500))

