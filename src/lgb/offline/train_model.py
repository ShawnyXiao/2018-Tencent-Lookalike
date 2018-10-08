
# coding: utf-8

# In[2]:


from __future__ import division
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.cross_validation import KFold
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
from multiprocessing import cpu_count
import gc
import warnings
warnings.filterwarnings('ignore')


# In[3]:


# Constants define
ROOT_PATH = '/home/kaholiu/xiaoxy/2018-Tencent-Lookalike/'
ONLINE = 0


# In[4]:


target = 'label'
train_len = 45539700  # 8798814
test1_len = 11729073  # 2265989
test2_len = 11727304  # 2265879
positive_num = 2182403  # 421961


# In[ ]:


########################################### Helper function ###########################################


# In[5]:


def log(info):
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' ' + str(info))


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
    result.to_csv(ROOT_PATH + 'data/output/sub/lgb/' + name + '.csv', index=False, sep=',')
    return result


# In[ ]:


########################################### Read data ###########################################


# In[7]:


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
# df_aidinter = cPickle.load(open(ROOT_PATH + 'data/output/lgb/final/feat/train+test1+test2/all(aidinter).p', 'rb'))
# df_adcatinter = cPickle.load(open(ROOT_PATH + 'data/output/lgb/final/feat/train+test1+test2/all(adcatinter).p', 'rb'))
# df_ageinter = cPickle.load(open(ROOT_PATH + 'data/output/lgb/final/feat/train+test1+test2/all(ageinter).p', 'rb'))

predictors_stat = list(df_stat.columns.values)
predictors_aduser = list(df_aduser.columns.values)
predictors_userad = list(df_userad.columns.values)
# predictors_aidinter = list(df_aidinter.columns.values)
# predictors_adcatinter = list(df_adcatinter.columns.values)
# predictors_ageinter = list(df_ageinter.columns.values)
# predictors_allid = ['aid', 'uid', 'LBS', 'age', 'carrier', 'consumptionAbility', 'education', 'gender', 'house', 'os', 'ct', 'advertiserId', 'campaignId', 'creativeId', 'adCategoryId', 'productId', 'productType']
df = pd.concat([df_basic, df_stat, df_aduser, df_userad], axis=1)
del df_basic, df_stat, df_aduser, df_userad; gc.collect()

# cut id count < x
# for i in ['LBS']:
#     df[i] = df[i].apply(lambda x: x + 1)
#     tmp = df.groupby(i).size().reset_index()
#     tmp.columns = ['id', 'count']
#     id_list = []
#     for _, row in tmp.iterrows():
#         if (row['count'] <= 5):
#             id_list.append(row['id'])
#     df[i] = df[i].apply(lambda x: x if x not in id_list else 0)

# Split dataset for local
train, test, _, _ = train_test_split(df, df[target], train_size=0.8, random_state=7, stratify=df[target])

predictors = []
predictors.append('creativeSize')
predictors += predictors_stat + predictors_aduser + predictors_userad
train_x = train[predictors]
test_x = test[predictors]
train_y = train[target]
test_y = test[target]

train_cv = sparse.load_npz('%sdata/output/lgb/final/feat/train+test1+test2/train(cv)(online=%s).npz' % (ROOT_PATH, str(ONLINE)))
test_cv = sparse.load_npz('%sdata/output/lgb/final/feat/train+test1+test2/test(cv)(online=%s).npz' % (ROOT_PATH, str(ONLINE)))
train_x = sparse.hstack((train_x, train_cv)); del train_cv; gc.collect()
test_x = sparse.hstack((test_x, test_cv)); del test_cv; gc.collect()


# In[ ]:


########################################### LigthGBM ###########################################


# In[14]:


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
    # 'device': 'gpu',
    # 'categorical_feature': ','.join([str(i) for i in range(0, 17)]),
}


# In[10]:


log('Now begin')
iterations_lgb, best_score_lgb, model_lgb_cv = lgb_cv(train_x, train_y, test_x, test_y, params_lgb, config_lgb['folds'], config_lgb['rounds'])


# In[42]:


log('Now begin')  # aidinter+aidinter2+aidinter3+adcatinter+ageinter
iterations_lgb, best_score_lgb, model_lgb_cv = lgb_cv(train_x, train_y, test_x, test_y, params_lgb, config_lgb['folds'], config_lgb['rounds'])


# In[45]:


log('Now begin')  # aidinter+aidinter2+aidinter3+adcatinter+ageinter+prodinter
iterations_lgb, best_score_lgb, model_lgb_cv = lgb_cv(train_x, train_y, test_x, test_y, params_lgb, config_lgb['folds'], config_lgb['rounds'])


# In[59]:


log('Now begin')  # aidinter+aidinter2+aidinter3+adcatinter+ageinter+aduser
iterations_lgb, best_score_lgb, model_lgb_cv = lgb_cv(train_x, train_y, test_x, test_y, params_lgb, config_lgb['folds'], config_lgb['rounds'])
log('Done')


# In[77]:


log('Now begin')  # aidinter+aidinter2+aidinter3+adcatinter+ageinter+aduser+allid
iterations_lgb, best_score_lgb, model_lgb_cv = lgb_cv(train_x, train_y, test_x, test_y, params_lgb, config_lgb['folds'], config_lgb['rounds'])
log('Done')  # no allid categorical_feature


# In[79]:


log('Now begin')  # allid aidinter+aidinter2+aidinter3+adcatinter+ageinter+aduser
iterations_lgb, best_score_lgb, model_lgb_cv = lgb_cv(train_x, train_y, test_x, test_y, params_lgb, config_lgb['folds'], config_lgb['rounds'])
log('Done')  # allid categorical_feature


# In[198]:


log('Now begin')  # aidinter+aidinter2+aidinter3+adcatinter+ageinter+aduser+allid(LBS_cut)
iterations_lgb, best_score_lgb, model_lgb_cv = lgb_cv(train_x, train_y, test_x, test_y, params_lgb, config_lgb['folds'], config_lgb['rounds'])
log('Done')


# In[200]:


log('Now begin')  # aidinter+aidinter2+aidinter3+adcatinter+ageinter+aduser+allid(LBS_cut) GPU
iterations_lgb, best_score_lgb, model_lgb_cv = lgb_cv(train_x, train_y, test_x, test_y, params_lgb, config_lgb['folds'], config_lgb['rounds'])
log('Done')


# In[9]:


log('Now begin')  # aidinter+aidinter2+aidinter3+adcatinter+ageinter+aduser+allid(LBS_cut)+userad
iterations_lgb, best_score_lgb, model_lgb_cv = lgb_cv(train_x, train_y, test_x, test_y, params_lgb, config_lgb['folds'], config_lgb['rounds'])
log('Done')


# In[ ]:


########################################### Final ###########################################


# In[12]:


log('Now begin')  # stat
iterations_lgb, best_score_lgb, model_lgb_cv = lgb_cv(train_x, train_y, test_x, test_y, params_lgb, config_lgb['folds'], config_lgb['rounds'])
log('Done')


# In[10]:


log('Now begin')  # stat+cv
iterations_lgb, best_score_lgb, model_lgb_cv = lgb_cv(train_x, train_y, test_x, test_y, params_lgb, config_lgb['folds'], config_lgb['rounds'])
log('Done')


# In[10]:


log('Now begin')  # stat+cv+aduser
iterations_lgb, best_score_lgb, model_lgb_cv = lgb_cv(train_x, train_y, test_x, test_y, params_lgb, config_lgb['folds'], config_lgb['rounds'])
log('Done')


# In[12]:


log('Now begin')  # stat+cv+aduser+userad
iterations_lgb, best_score_lgb, model_lgb_cv = lgb_cv(train_x, train_y, test_x, test_y, params_lgb, config_lgb['folds'], config_lgb['rounds'])
log('Done')


# In[9]:


log('Now begin')  # stat+cv+aduser+userad+adage
iterations_lgb, best_score_lgb, model_lgb_cv = lgb_cv(train_x, train_y, test_x, test_y, params_lgb, config_lgb['folds'], config_lgb['rounds'])
log('Done')


# In[ ]:


########################################### 1：3 ###########################################


# In[9]:


log('Now begin')  # stat+cv+aduser+userad lr=0.1
model_lgb_cv, iterations_lgb, best_score_lgb = lgb_cv(train_x, train_y, test_x, test_y, params_lgb, config_lgb['folds'], config_lgb['rounds'])
log('Done')


# In[15]:


log('Now begin')  # stat+cv+aduser+userad lr=0.01
model_lgb_cv, iterations_lgb, best_score_lgb = lgb_cv(train_x, train_y, test_x, test_y, params_lgb, config_lgb['folds'], config_lgb['rounds'])
log('Done')


# In[17]:


# Save model
model_lgb_cv.save_model(ROOT_PATH + 'data/output/lgb/final/model/lgb-cv-%d(r%d).txt' % (7517, 3350))

