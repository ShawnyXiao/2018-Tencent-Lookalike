
# coding: utf-8

# In[ ]:


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


# In[ ]:


# Constants define
ROOT_PATH = '/home/kaholiu/xiaoxy/2018-Tencent-Lookalike/'
ONLINE = 0


# In[ ]:


target = 'label'
train_len = 45539700  # 8798814
test1_len = 11729073  # 2265989
test2_len = 11727304  # 2265879
positive_num = 2182403  # 421961


# In[ ]:


########################################### Helper function ###########################################


# In[ ]:


def log(info):
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' ' + str(info))


# In[ ]:


def merge_count(df, columns_groupby, new_column_name, type='uint64'):
    add = pd.DataFrame(df.groupby(columns_groupby).size()).reset_index()
    add.columns = columns_groupby + [new_column_name]
    df = df.merge(add, on=columns_groupby, how="left"); del add; gc.collect()
    df[new_column_name] = df[new_column_name].astype(type)
    return df


# In[ ]:


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


# In[ ]:


def down_sample(df, df_feat):
    df_majority = df_feat[df[target]==0]
    df_minority = df_feat[df[target]==1]
    df_majority_downsampled = resample(df_majority,
                                     replace=False,  # sample without replacement
                                     n_samples=positive_num*3,  # to match minority class
                                     random_state=7)  # reproducible results
    df_downsampled = pd.concat([df_majority_downsampled, df_minority])
    del df_majority, df_minority, df_majority_downsampled
    return df_downsampled


# In[ ]:


########################################### Read data ###########################################


# In[ ]:


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


# In[ ]:


onehot_feature = ['LBS', 'age', 'carrier', 'consumptionAbility', 'education', 'gender', 'house', 'os', 'ct', 'advertiserId', 'campaignId', 'creativeId', 'adCategoryId', 'productId', 'productType']
for feature in onehot_feature:
    log(feature)
    try:
        df[feature] = LabelEncoder().fit_transform(df[feature].apply(int))
    except:
        df[feature] = LabelEncoder().fit_transform(df[feature])


# In[ ]:


vector_feature = ['appIdAction', 'appIdInstall', 'interest1', 'interest2', 'interest3', 'interest4', 'interest5', 'kw1', 'kw2', 'kw3', 'marriageStatus', 'topic1', 'topic2', 'topic3']
for feature in vector_feature:
    log(feature)
    df[feature] = preprocess_word(df[feature].values)


# In[ ]:


# Save aid+uid+label+creativeSize columns
df_downsampled = down_sample(df, df[['aid', 'uid', 'label', 'creativeSize']])
cPickle.dump(df_downsampled, open(ROOT_PATH + 'data/output/lgb/final/feat/train+test1+test2/all(basic).p', 'wb')); del df_downsampled; gc.collect()


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
df_downsampled = down_sample(df, df[predictors_stat1])
df.drop(predictors_stat1, axis=1, inplace=True); gc.collect()
cPickle.dump(df_downsampled, open(ROOT_PATH + 'data/output/lgb/final/feat/train+test1+test2/all(stat1).p', "wb")); del df_downsampled; gc.collect()


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
df_downsampled = down_sample(df, df[predictors_stat2])
df.drop(predictors_stat2, axis=1, inplace=True); gc.collect()
cPickle.dump(df_downsampled, open(ROOT_PATH + 'data/output/lgb/final/feat/train+test1+test2/all(stat2).p', "wb")); del df_downsampled; gc.collect()


# In[ ]:


# Stat features
predictors_stat3 = []

vector_feature = ['appIdAction', 'appIdInstall', 'interest1', 'interest2', 'interest3', 'interest4', 'interest5', 'kw1', 'kw2', 'kw3', 'marriageStatus', 'topic1', 'topic2', 'topic3']
for feature in vector_feature:
    log(feature)
    df['len_' + feature] = [0 if var == 'W-1' else len(var.split()) for var in df[feature].values]
    predictors_stat3.append('len_' + feature)

# Save features
df_downsampled = down_sample(df, df[predictors_stat3])
df.drop(predictors_stat3, axis=1, inplace=True); gc.collect()
cPickle.dump(df_downsampled, open(ROOT_PATH + 'data/output/lgb/final/feat/train+test1+test2/all(stat3).p', "wb")); del df_downsampled; gc.collect()


# In[ ]:


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
cPickle.dump(df_stat[predictors_stat4], open(ROOT_PATH + 'data/output/lgb/final/feat/train+test1+test2/all(stat4).p', "wb")); del df_stat; gc.collect()


# In[ ]:


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
df_downsampled = down_sample(df, df_aduser)
cPickle.dump(df_downsampled, open(ROOT_PATH + 'data/output/lgb/final/feat/train+test1+test2/all(aduser1).p', "wb")); del df_downsampled, df_aduser; gc.collect()


# In[ ]:


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
df_downsampled = down_sample(df, df_aduser)
cPickle.dump(df_downsampled, open(ROOT_PATH + 'data/output/lgb/final/feat/train+test1+test2/all(aduser2).p', "wb")); del df_downsampled, df_aduser; gc.collect()


# In[ ]:


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
df_downsampled = down_sample(df, df_userad)
cPickle.dump(df_downsampled, open(ROOT_PATH + 'data/output/lgb/final/feat/train+test1+test2/all(userad1).p', "wb")); del df_downsampled, df_userad; gc.collect()


# In[ ]:


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
df_downsampled = down_sample(df, df_userad)
cPickle.dump(df_downsampled, open(ROOT_PATH + 'data/output/lgb/final/feat/train+test1+test2/all(userad2).p', "wb")); del df_downsampled, df_userad; gc.collect()


# In[ ]:


# Split dataset for local
df_downsampled = down_sample(df, df)
train, test, _, _ = train_test_split(df_downsampled, df_downsampled[target], train_size=0.8, random_state=7, stratify=df_downsampled[target]); del df_downsampled; gc.collect()

# Construct count vector features
train_cv = train[[]]
test_cv = test[[]]
vector_feature = ['appIdAction', 'appIdInstall', 'interest1', 'interest2', 'interest3', 'interest4', 'interest5', 'kw1', 'kw2', 'kw3', 'marriageStatus', 'topic1', 'topic2', 'topic3']
cv = CountVectorizer()
for feature in vector_feature:
    log(feature)
    cv.fit(df[feature])
    train_cv_tmp = cv.transform(train[feature])
    test_cv_tmp = cv.transform(test[feature])
    train_cv = sparse.hstack((train_cv, train_cv_tmp))
    test_cv = sparse.hstack((test_cv, test_cv_tmp))
    del train_cv_tmp, test_cv_tmp; gc.collect()

# Save all count-vector features
log('Count vector shape for train: ' + str(train_cv.shape))
log('Count vector shape for test: ' + str(test_cv.shape))
sparse.save_npz('%sdata/output/lgb/final/feat/train+test1+test2/train(cv)(online=%s).npz' % (ROOT_PATH, str(ONLINE)), train_cv)
sparse.save_npz('%sdata/output/lgb/final/feat/train+test1+test2/test(cv)(online=%s).npz' % (ROOT_PATH, str(ONLINE)), test_cv)
del train_cv, test_cv, train, test; gc.collect()


# In[ ]:


gb_list = ['carrier', 'consumptionAbility', 'house', 'os', 'LBS', 'ct', 'age', 'education', 'gender']

predictors_aidinter = []
for i in gb_list:
    cola = 'aid'
    colb = i
    df[cola + '_' + colb] = df[cola].astype(str) + '_' + df[colb].astype(str)
    try:
        df[cola + '_' + colb] = LabelEncoder().fit_transform(df[cola + '_' + colb].apply(int))
    except:
        df[cola + '_' + colb] = LabelEncoder().fit_transform(df[cola + '_' + colb])
    predictors_aidinter.append(cola + '_' + colb)
    
# Save features
df_downsampled = down_sample(df, df[predictors_aidinter])
df.drop(predictors_aidinter, axis=1, inplace=True); gc.collect()
cPickle.dump(df_downsampled, open(ROOT_PATH + 'data/output/lgb/final/feat/train+test1+test2/all(aidinter).p', "wb")); del df_downsampled; gc.collect()


# In[ ]:


gb_list = ['age', 'carrier', 'consumptionAbility', 'education', 'gender', 'house', 'os', 'ct']

predictors_adcatinter = []
for i in gb_list:
    cola = 'adCategoryId'
    colb = i
    df[cola + '_' + colb] = df[cola].astype(str) + '_' + df[colb].astype(str)
    try:
        df[cola + '_' + colb] = LabelEncoder().fit_transform(df[cola + '_' + colb].apply(int))
    except:
        df[cola + '_' + colb] = LabelEncoder().fit_transform(df[cola + '_' + colb])
    predictors_adcatinter.append(cola + '_' + colb)
    
# Save features
df_downsampled = down_sample(df, df[predictors_adcatinter])
df.drop(predictors_adcatinter, axis=1, inplace=True); gc.collect()
cPickle.dump(df_downsampled, open(ROOT_PATH + 'data/output/lgb/final/feat/train+test1+test2/all(adcatinter).p', "wb")); del df_downsampled; gc.collect()


# In[ ]:


gb_list = ['adCategoryId', 'productId', 'productType']

predictors_ageinter = []
for i in gb_list:
    cola = 'age'
    colb = i
    df[cola + '_' + colb] = df[cola].astype(str) + '_' + df[colb].astype(str)
    try:
        df[cola + '_' + colb] = LabelEncoder().fit_transform(df[cola + '_' + colb].apply(int))
    except:
        df[cola + '_' + colb] = LabelEncoder().fit_transform(df[cola + '_' + colb])
    predictors_ageinter.append(cola + '_' + colb)
    
# Save features
df_downsampled = down_sample(df, df[predictors_ageinter])
df.drop(predictors_ageinter, axis=1, inplace=True); gc.collect()
cPickle.dump(df_downsampled, open(ROOT_PATH + 'data/output/lgb/final/feat/train+test1+test2/all(ageinter).p', "wb")); del df_downsampled; gc.collect()


# In[ ]:


gb_list = ['age', 'carrier', 'consumptionAbility', 'education', 'gender', 'house', 'os', 'ct']

predictors_prodinter = []
for k in ['productId', 'productType']:
    for i in gb_list:
        cola = k
        colb = i
        df[cola + '_' + colb] = df[cola].astype(str) + '_' + df[colb].astype(str)
        try:
            df[cola + '_' + colb] = LabelEncoder().fit_transform(df[cola + '_' + colb].apply(int))
        except:
            df[cola + '_' + colb] = LabelEncoder().fit_transform(df[cola + '_' + colb])
        predictors_prodinter.append(cola + '_' + colb)
    
# Save features
df_downsampled = down_sample(df, df[predictors_prodinter])
df.drop(predictors_prodinter, axis=1, inplace=True); gc.collect()
cPickle.dump(df_downsampled, open(ROOT_PATH + 'data/output/lgb/final/feat/train+test1+test2/all(prodinter).p', "wb")); del df_downsampled; gc.collect()

