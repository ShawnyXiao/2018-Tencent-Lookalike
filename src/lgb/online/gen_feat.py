
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


def merge_count(df, columns_groupby, new_column_name, type='uint64'):
    add = pd.DataFrame(df.groupby(columns_groupby).size()).reset_index()
    add.columns = columns_groupby + [new_column_name]
    df = df.merge(add, on=columns_groupby, how="left"); del add; gc.collect()
    df[new_column_name] = df[new_column_name].astype(type)
    return df


# In[7]:


def preprocess_word(texts):
    pre_texts = []
    for text in texts:
        words = text.split()
        pre_words = []
        for word in words:
            pre_words.append('W' + word)
        pre_text = ' '.join(pre_words)
        pre_texts.append(pre_text)
    return pre_texts


# In[8]:


def get_index_of_one_sixth_of_train(train_y):
    skf = StratifiedKFold(n_splits=6)
    for i, (_, test_index) in enumerate(skf.split(np.zeros(len(train_y)), train_y)):
        if i == 1:  # !
            return test_index


# In[ ]:


########################################### Read data ###########################################


# In[9]:


ad_feature = pd.read_csv(ROOT_PATH + 'data/input/final/adFeature.csv', header=0, sep=',')
user_feature = pd.read_csv(ROOT_PATH + 'data/input/final/userFeature.csv', header=0, sep=',')
train = pd.read_csv(ROOT_PATH + 'data/input/final/train.csv', header=0, sep=',')
test1 = pd.read_csv(ROOT_PATH + 'data/input/final/test1.csv', header=0, sep=',')
test2 = pd.read_csv(ROOT_PATH + 'data/input/final/test2.csv', header=0, sep=',')
test = test1.append(test2).reset_index(drop=True); del test1, test2; gc.collect()

df = train.append(test).reset_index(drop=True)
df = df.merge(ad_feature, on='aid', how='left')
df = df.merge(user_feature, on='uid', how='left')
del train, test, ad_feature, user_feature; gc.collect()

df.loc[df[target] == -1, target] = 0
df.loc[train_len:, target] = -1
df[target] = df[target].astype(int)

df = df.fillna('-1')


# In[ ]:


########################################### Preprocess ###########################################


# In[10]:


onehot_feature = ['LBS', 'age', 'carrier', 'consumptionAbility', 'education', 'gender', 'house', 'os', 'ct', 'advertiserId', 'campaignId', 'creativeId', 'adCategoryId', 'productId', 'productType']
for feature in onehot_feature:
    log(feature)
    try:
        df[feature] = LabelEncoder().fit_transform(df[feature].apply(int))
    except:
        df[feature] = LabelEncoder().fit_transform(df[feature])


# In[11]:


vector_feature = ['appIdAction', 'appIdInstall', 'interest1', 'interest2', 'interest3', 'interest4', 'interest5', 'kw1', 'kw2', 'kw3', 'marriageStatus', 'topic1', 'topic2', 'topic3']
for feature in vector_feature:
    log(feature)
    df[feature] = preprocess_word(df[feature].values)


# In[ ]:


# Save aid+uid+label+creativeSize columns
# sample_index = get_index_of_one_sixth_of_train(df.loc[:(train_len-1), target])
cPickle.dump(df.loc[train_len:, ['aid', 'uid', 'label', 'creativeSize']], open(ROOT_PATH + 'data/output/lgb/final/feat/train+test1+test2/all(basic).p', 'wb')); gc.collect()


# In[ ]:


########################################### Feature engineer ###########################################


# In[ ]:


log('Before feature engineer')
log('Num of columns: ' + str(len(df.columns)))
log('columns: ' + str(df.columns))


# In[ ]:


# Stat features
predictors_stat1 = []

gb_list = ['uid', 'aid', 'LBS', 'age', 'carrier', 'consumptionAbility', 'education', 'gender', 'house', 'os', 'ct', 'advertiserId', 'campaignId', 'creativeId', 'adCategoryId', 'productId', 'productType', 'marriageStatus']
for i in gb_list:
    log(i)
    df = merge_count(df, [i], 'count_gb_' + i, 'uint32')
    predictors_stat1.append('count_gb_' + i)

gb_list = ['LBS', 'age', 'carrier', 'consumptionAbility', 'education', 'gender', 'house', 'os', 'ct']
for i in gb_list:
    log('aid_' + i)
    df = merge_count(df, ['aid', i], 'count_gb_aid_' + i, 'uint32')
    predictors_stat1.append('count_gb_aid_' + i)
    
# Save features
cPickle.dump(df.loc[:, predictors_stat1], open(ROOT_PATH + 'data/output/lgb/final/feat/train+test1+test2/all(stat1).p', "wb")); gc.collect()
df.drop(predictors_stat1, axis=1, inplace=True); gc.collect()


# In[ ]:


# Stat features
predictors_stat2 = []

gb_user = ['uid', 'age', 'LBS']
gb_list = ['advertiserId', 'campaignId', 'creativeId', 'adCategoryId', 'productId', 'productType']
for u in gb_user:
    for i in gb_list:
        log(u + '_' + i)
        df = merge_count(df, [u, i], 'count_gb_%s_%s' % (u, i), 'uint32')
        predictors_stat2.append('count_gb_%s_%s' % (u, i))

# Save features
cPickle.dump(df.loc[:, predictors_stat2], open(ROOT_PATH + 'data/output/lgb/final/feat/train+test1+test2/all(stat2).p', "wb")); gc.collect()
df.drop(predictors_stat2, axis=1, inplace=True); gc.collect()


# In[ ]:


# Stat features
predictors_stat3 = []

vector_feature = ['appIdAction', 'appIdInstall', 'interest1', 'interest2', 'interest3', 'interest4', 'interest5', 'kw1', 'kw2', 'kw3', 'marriageStatus', 'topic1', 'topic2', 'topic3']
for feature in vector_feature:
    log(feature)
    df['len_' + feature] = [0 if var == 'W-1' else len(var.split()) for var in df[feature].values]
    predictors_stat3.append('len_' + feature)

# Save features
cPickle.dump(df.loc[:, predictors_stat3], open(ROOT_PATH + 'data/output/lgb/final/feat/train+test1+test2/all(stat3).p', "wb")); gc.collect()
df.drop(predictors_stat3, axis=1, inplace=True); gc.collect()


# In[17]:


# Stat features
df_stat1 = cPickle.load(open(ROOT_PATH + 'data/output/lgb/final/feat/train+test1+test2/all(stat1).p', 'rb'))
df_stat2 = cPickle.load(open(ROOT_PATH + 'data/output/lgb/final/feat/train+test1+test2/all(stat2).p', 'rb'))
df_stat = pd.concat([df_stat1, df_stat2], axis=1); del df_stat1, df_stat2; gc.collect()

predictors_stat4 = []

gb_list = ['LBS', 'age', 'carrier', 'consumptionAbility', 'education', 'gender', 'house', 'os', 'ct']
for i in gb_list:
    log('aid_' + i)
    df_stat['ratio_aid_%s_to_aid' % i] = df_stat['count_gb_aid_' + i].astype(float) / df_stat['count_gb_aid']
    df_stat['ratio_aid_%s_to_%s' % (i, i)] = df_stat['count_gb_aid_' + i].astype(float) / df_stat['count_gb_' + i]
    predictors_stat4.append('ratio_aid_%s_to_aid' % i)
    predictors_stat4.append('ratio_aid_%s_to_%s' % (i, i))

gb_user = ['uid', 'age', 'LBS']
gb_list = ['advertiserId', 'campaignId', 'creativeId', 'adCategoryId', 'productId', 'productType']
for u in gb_user:
    for i in gb_list:
        log(u + '_' + i)
        df_stat['ratio_%s_%s_to_%s' % (u, i, u)] = df_stat['count_gb_%s_%s' % (u, i)] / df_stat['count_gb_' + u]
        df_stat['ratio_%s_%s_to_%s' % (u, i, i)] = df_stat['count_gb_%s_%s' % (u, i)] / df_stat['count_gb_' + i]
        predictors_stat4.append('ratio_%s_%s_to_%s' % (u, i, u))
        predictors_stat4.append('ratio_%s_%s_to_%s' % (u, i, i))

# Save features
# cPickle.dump(df_stat[predictors_stat4], open(ROOT_PATH + 'data/output/lgb/final/feat/train+test1+test2/all(stat4).p', "wb")); del df_stat; gc.collect()
cPickle.dump(df_stat.iloc[train_len:, predictors_stat4], open(ROOT_PATH + 'data/output/lgb/final/feat/train+test1+test2/test(stat4).p', "wb")); del df_stat; gc.collect()


# In[26]:


cPickle.dump(df_stat.iloc[train_len:, :], open(ROOT_PATH + 'data/output/lgb/final/feat/train+test1+test2/test(stat1+stat2+stat4).p', "wb")); del df_stat; gc.collect()


# In[12]:


gb_list = ['age', 'carrier', 'consumptionAbility', 'education']

predictors_aduser = []
df_aduser = df[[]]
for i in gb_list:
    log(i)
    column_name = i
    df_onehot = pd.get_dummies(df[column_name], prefix=column_name)
    df_tmp = df_onehot.groupby(df.aid).transform(np.mean)
    df_tmp.columns = [i + '_gb_aid' for i in df_tmp.columns]
    df_aduser = pd.concat([df_aduser, df_tmp], axis=1)
    predictors_aduser += list(df_tmp.columns.values)
    df_tmp = df_tmp.groupby(df.uid).transform(np.mean)
    df_tmp.columns = [i + '_gb_uid' for i in df_tmp.columns]
    df_aduser = pd.concat([df_aduser, df_tmp], axis=1)
    predictors_aduser += list(df_tmp.columns.values)
log(predictors_aduser)

# Save features
# cPickle.dump(df_aduser.loc[:, predictors_aduser], open(ROOT_PATH + 'data/output/lgb/final/feat/train+test1+test2/all(aduser1).p', "wb")); del df_aduser; gc.collect()
cPickle.dump(df_aduser.loc[train_len:, predictors_aduser], open(ROOT_PATH + 'data/output/lgb/final/feat/train+test1+test2/test(aduser1).p', "wb")); del df_aduser; gc.collect()


# In[13]:


gb_list = ['gender', 'house', 'os', 'productType']  # 'adCategoryId', 'productId'

predictors_aduser = []
df_aduser = df[[]]
for i in gb_list:
    log(i)
    column_name = i
    df_onehot = pd.get_dummies(df[column_name], prefix=column_name)
    df_tmp = df_onehot.groupby(df.aid).transform(np.mean)
    df_tmp.columns = [i + '_gb_aid' for i in df_tmp.columns]
    df_aduser = pd.concat([df_aduser, df_tmp], axis=1)
    predictors_aduser += list(df_tmp.columns.values)
    df_tmp = df_tmp.groupby(df.uid).transform(np.mean)
    df_tmp.columns = [i + '_gb_uid' for i in df_tmp.columns]
    df_aduser = pd.concat([df_aduser, df_tmp], axis=1)
    predictors_aduser += list(df_tmp.columns.values)
log(predictors_aduser)

# Save features
# cPickle.dump(df_aduser.loc[:, predictors_aduser], open(ROOT_PATH + 'data/output/lgb/final/feat/train+test1+test2/all(aduser2).p', "wb")); del df_aduser; gc.collect()
cPickle.dump(df_aduser.loc[train_len:, predictors_aduser], open(ROOT_PATH + 'data/output/lgb/final/feat/train+test1+test2/test(aduser2).p', "wb")); del df_aduser; gc.collect()


# In[14]:


gb_list = ['age', 'carrier', 'consumptionAbility', 'education']

predictors_userad = []
df_userad = df[[]]
for i in gb_list:
    log(i)
    column_name = i
    df_onehot = pd.get_dummies(df[column_name], prefix=column_name)
    df_tmp = df_onehot.groupby(df.uid).transform(np.mean)
    df_tmp.columns = [i + '_gb_uid' for i in df_tmp.columns]
    df_userad = pd.concat([df_userad, df_tmp], axis=1)
    predictors_userad += list(df_tmp.columns.values)
    df_tmp = df_tmp.groupby(df.aid).transform(np.mean)
    df_tmp.columns = [i + '_gb_aid' for i in df_tmp.columns]
    df_userad = pd.concat([df_userad, df_tmp], axis=1)
    predictors_userad += list(df_tmp.columns.values)
log(predictors_userad)

# Save features
# cPickle.dump(df_userad.loc[:, predictors_userad], open(ROOT_PATH + 'data/output/lgb/final/feat/train+test1+test2/all(userad1).p', "wb")); del df_userad; gc.collect()
cPickle.dump(df_userad.loc[train_len:, predictors_userad], open(ROOT_PATH + 'data/output/lgb/final/feat/train+test1+test2/test(userad1).p', "wb")); del df_userad; gc.collect()


# In[15]:


gb_list = ['gender', 'house', 'os', 'productType']  # 'adCategoryId', 'productId'

predictors_userad = []
df_userad = df[[]]
for i in gb_list:
    log(i)
    column_name = i
    df_onehot = pd.get_dummies(df[column_name], prefix=column_name)
    df_tmp = df_onehot.groupby(df.uid).transform(np.mean)
    df_tmp.columns = [i + '_gb_uid' for i in df_tmp.columns]
    df_userad = pd.concat([df_userad, df_tmp], axis=1)
    predictors_userad += list(df_tmp.columns.values)
    df_tmp = df_tmp.groupby(df.aid).transform(np.mean)
    df_tmp.columns = [i + '_gb_aid' for i in df_tmp.columns]
    df_userad = pd.concat([df_userad, df_tmp], axis=1)
    predictors_userad += list(df_tmp.columns.values)
log(predictors_userad)

# Save features
# cPickle.dump(df_userad.loc[:, predictors_userad], open(ROOT_PATH + 'data/output/lgb/final/feat/train+test1+test2/all(userad2).p', "wb")); del df_userad; gc.collect()
cPickle.dump(df_userad.loc[train_len:, predictors_userad], open(ROOT_PATH + 'data/output/lgb/final/feat/train+test1+test2/test(userad2).p', "wb")); del df_userad; gc.collect()


# In[16]:


# Split dataset for online
# train = df.loc[:, :]
test = df.loc[train_len:, :]

# Construct count vector features
# train_cv = train[[]]
test_cv = test[[]]
vector_feature = ['appIdAction', 'appIdInstall', 'interest1', 'interest2', 'interest3', 'interest4', 'interest5', 'kw1', 'kw2', 'kw3', 'marriageStatus', 'topic1', 'topic2', 'topic3']
cv = CountVectorizer()
for feature in vector_feature:
    log(feature)
    cv.fit(df[feature])
#     train_cv_tmp = cv.transform(train[feature])
    test_cv_tmp = cv.transform(test[feature])
#     train_cv = sparse.hstack((train_cv, train_cv_tmp))
    test_cv = sparse.hstack((test_cv, test_cv_tmp))
#     del train_cv_tmp, test_cv_tmp; gc.collect()
    del test_cv_tmp; gc.collect()

# Save all count-vector features
# log('Count vector shape for train: ' + str(train_cv.shape))
log('Count vector shape for test: ' + str(test_cv.shape))
# sparse.save_npz('%sdata/output/lgb/final/feat/train+test1+test2/train(cv)(online=%s).npz' % (ROOT_PATH, str(ONLINE)), train_cv)
sparse.save_npz('%sdata/output/lgb/final/feat/train+test1+test2/test(cv)(online=%s).npz' % (ROOT_PATH, str(ONLINE)), test_cv)
# del train_cv, test_cv, train, test; gc.collect()
del test_cv, test; gc.collect()

