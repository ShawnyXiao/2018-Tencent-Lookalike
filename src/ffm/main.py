
# coding: utf-8

# In[1]:


from __future__ import division
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from scipy import sparse
import lightgbm as lgb
import xlearn as xl
import cPickle
import time
import gc
import warnings
warnings.filterwarnings('ignore')


# In[2]:


# Constants define
ROOT_PATH = '/home/kaholiu/xiaoxy/2018-Tencent-Lookalike/'
TRAIN_LEN = 8798814


# In[3]:


class FFMFormat:
    def __init__(self, vector_feat, one_hot_feat, continous_feat):
        self.field_index_ = None
        self.feature_index_ = None
        self.vector_feat = vector_feat
        self.one_hot_feat = one_hot_feat
        self.continous_feat = continous_feat

    def get_params(self):
        pass

    def set_params(self, **parameters):
        pass

    def fit(self, df, y=None):
        self.field_index_ = {col: i for i, col in enumerate(df.columns)}
        self.feature_index_ = dict()
        last_idx = 0
        for col in df.columns:
            if col in self.one_hot_feat:
                print('Fitting column: ' + col)
                df[col] = df[col].astype('int')
                vals = np.unique(df[col])
                for val in vals:
                    if val == -1: continue
                    name = '{}_{}'.format(col, val)
                    if name not in self.feature_index_:
                        self.feature_index_[name] = last_idx
                        last_idx += 1
            elif col in self.vector_feat:
                print('Fitting column: ' + col)
                vals = []
                for data in df[col].apply(str):
                    if data != '-1':
                        for word in data.strip().split(' '):
                            vals.append(word)
                vals = np.unique(vals)
                for val in vals:
                    if val == '-1': continue
                    name = '{}_{}'.format(col, val)
                    if name not in self.feature_index_:
                        self.feature_index_[name] = last_idx
                        last_idx += 1
            else:
                self.feature_index_[col] = last_idx
                last_idx += 1
            if col in aid_inter_oh_feature:
                print('Fitting column (inter aid): ' + col)
                self.field_index_['aid_{}'.format(col)] = len(self.field_index_)
                df['aid'] = df['aid'].astype('int')
                df[col] = df[col].astype('int')
                vals = df[['aid', col]].drop_duplicates()
                for idx, val in vals.iterrows():
                    if val[col] == -1: continue
                    name = 'aid_{}_{}_{}'.format(val['aid'], col, val[col])
                    if name not in self.feature_index_:
                        self.feature_index_[name] = last_idx
                        last_idx += 1
            elif col in aid_inter_vec_feature:
                print('Fitting column (inter aid): ' + col)
                self.field_index_['aid_{}'.format(col)] = len(self.field_index_)
                df['aid'] = df['aid'].astype('int')
                df[col] = df[col].astype(str)
                vals = df[['aid', col]].drop_duplicates()
                for idx, val in vals.iterrows():
                    if val[col] == '-1': continue
                    for word in val[col].strip().split(' '):
                        name = 'aid_{}_{}_{}'.format(val['aid'], col, word)
                        if name not in self.feature_index_:
                            self.feature_index_[name] = last_idx
                            last_idx += 1
        return self

    def fit_transform(self, df, y=None):
        self.fit(df, y)
        return self.transform(df)

    def transform_row_(self, row):
        ffm = []
        for col, val in row.loc[row != 0].to_dict().items():
            if col in self.one_hot_feat:
                name = '{}_{}'.format(col, val)
                if name in self.feature_index_:
                    ffm.append('{}:{}:1'.format(self.field_index_[col], self.feature_index_[name]))
            elif col in self.vector_feat:
                for word in str(val).split(' '):
                    name = '{}_{}'.format(col, word)
                    if name in self.feature_index_:
                        ffm.append('{}:{}:1'.format(self.field_index_[col], self.feature_index_[name]))
            elif col in self.continous_feat:
                if val != -1:
                    ffm.append('{}:{}:{}'.format(self.field_index_[col], self.feature_index_[col], val))
            if col in aid_inter_oh_feature:
                name = 'aid_{}_{}_{}'.format(row['aid'], col, val)
                if name in self.feature_index_:
                    ffm.append('{}:{}:1'.format(self.field_index_['aid_{}'.format(col)], self.feature_index_[name]))
            elif col in aid_inter_vec_feature:
                for word in str(val).split(' '):
                    name = 'aid_{}_{}_{}'.format(row['aid'], col, word)
                    if name in self.feature_index_:
                        ffm.append('{}:{}:1'.format(self.field_index_['aid_{}'.format(col)], self.feature_index_[name]))
        return ' '.join(ffm)

    def transform(self, df):
        return pd.Series({idx: self.transform_row_(row) for idx, row in df.iterrows()})


# In[4]:


def log(info):
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' ' + str(info))


def log_shape(train, test):
    log('Train data shape: %s' % str(train.shape))
    log('Test data shape: %s' % str(test.shape))
    

def save_obj(obj, name):
    with open(ROOT_PATH + 'data/output/feat/ffm/' + name + '.pkl', 'wb') as f:
        cPickle.dump(obj, f, cPickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(ROOT_PATH + 'data/output/feat/ffm/' + name + '.pkl', 'rb') as f:
        return cPickle.load(f)


# In[ ]:


########################################### Generate LibFFM data ###########################################


# In[5]:


log('Read data...')
dtypes = {
    'aid': 'uint16',
    'uid': 'uint32',
    'label': 'int8',
    'advertiserId': 'uint32',
    'campaignId': 'uint32',
    'creativeId': 'uint32',
    'creativeSize': 'uint8',
    'adCategoryId': 'uint16',
    'productId': 'uint16',
    'productType': 'uint8',
    'LBS': 'float16',  # 'uint16' bug: NaN
    'age': 'uint8',
    'carrier': 'uint8',
    'consumptionAbility': 'uint8',
    'education': 'uint8',
    'gender': 'uint8',
    'house': 'float16'  # 'uint8'  bug: NaN
}
ad_feature = pd.read_csv(ROOT_PATH + 'data/input/pre/adFeature.csv', header=0, sep=',', dtype=dtypes)
user_feature = pd.read_csv(ROOT_PATH + 'data/input/pre/userFeature.csv', header=0, sep=',', dtype=dtypes)
train = pd.read_csv(ROOT_PATH + 'data/input/pre/train.csv', header=0, sep=',', dtype=dtypes)
test = pd.read_csv(ROOT_PATH + 'data/input/pre/test1.csv', header=0, sep=',', dtype=dtypes)
log_shape(train, test)
log('Read data done!')


# In[6]:


log('Train append test...')
df = train.append(test).reset_index(drop=True)
df = df.merge(ad_feature, on='aid', how='left')
df = df.merge(user_feature, on='uid', how='left')
del ad_feature, user_feature
log('Train append test done!')

log('Transform label from -1 to 0 and NaN to -1...')
df.loc[df['label'] == -1, 'label'] = 0
df.loc[TRAIN_LEN:, 'label'] = -1
log('Transform label from -1 to 0 and NaN to -1 done!')

log('Fill NaN with \'-1\'...')
df = df.fillna('-1')
log('Fill NaN with \'-1\' done!')


# In[7]:


log('Concat stat features...')
df_stat = pd.read_csv(ROOT_PATH + 'data/output/feat/all(stat).csv', dtype=dtypes)
df = pd.concat([df, df_stat], axis=1).reset_index(drop=True); del df_stat; gc.collect()
stat_feature = ['count_gb_uid', 'count_gb_aid', 'count_gb_LBS', 'count_gb_age', 'count_gb_carrier', 'count_gb_consumptionAbility', 'count_gb_education', 'count_gb_gender', 'count_gb_house', 'count_gb_os', 'count_gb_ct', 'count_gb_advertiserId', 'count_gb_campaignId', 'count_gb_creativeId', 'count_gb_adCategoryId', 'count_gb_productId', 'count_gb_productType', 'count_gb_marriageStatus', 'count_gb_aid_age', 'count_gb_aid_gender', 'count_gb_aid_marriageStatus', 'count_gb_aid_education', 'count_gb_aid_consumptionAbility', 'count_gb_aid_LBS', 'count_gb_aid_ct', 'count_gb_aid_os', 'count_gb_aid_carrier', 'count_gb_aid_house', 'count_gb_uid_advertiserId', 'count_gb_uid_campaignId', 'count_gb_uid_creativeId', 'count_gb_uid_adCategoryId', 'count_gb_uid_productId', 'count_gb_uid_productType', 'count_gb_age_advertiserId', 'count_gb_age_campaignId', 'count_gb_age_creativeId', 'count_gb_age_adCategoryId', 'count_gb_age_productId', 'count_gb_age_productType', 'count_gb_gender_advertiserId', 'count_gb_gender_campaignId', 'count_gb_gender_creativeId', 'count_gb_gender_adCategoryId', 'count_gb_gender_productId', 'count_gb_gender_productType', 'count_gb_marriageStatus_advertiserId', 'count_gb_marriageStatus_campaignId', 'count_gb_marriageStatus_creativeId', 'count_gb_marriageStatus_adCategoryId', 'count_gb_marriageStatus_productId', 'count_gb_marriageStatus_productType', 'count_gb_education_advertiserId', 'count_gb_education_campaignId', 'count_gb_education_creativeId', 'count_gb_education_adCategoryId', 'count_gb_education_productId', 'count_gb_education_productType', 'count_gb_consumptionAbility_advertiserId', 'count_gb_consumptionAbility_campaignId', 'count_gb_consumptionAbility_creativeId', 'count_gb_consumptionAbility_adCategoryId', 'count_gb_consumptionAbility_productId', 'count_gb_consumptionAbility_productType', 'count_gb_LBS_advertiserId', 'count_gb_LBS_campaignId', 'count_gb_LBS_creativeId', 'count_gb_LBS_adCategoryId', 'count_gb_LBS_productId', 'count_gb_LBS_productType', 'count_gb_house_advertiserId', 'count_gb_house_campaignId', 'count_gb_house_creativeId', 'count_gb_house_adCategoryId', 'count_gb_house_productId', 'count_gb_house_productType', 'len_appIdAction', 'len_appIdInstall', 'len_interest1', 'len_interest2', 'len_interest3', 'len_interest4', 'len_interest5', 'len_kw1', 'len_kw2', 'len_kw3', 'len_topic1', 'len_topic2', 'len_topic3']
log('Concat stat features done!')


# In[8]:


one_hot_feature = ['LBS', 'age', 'carrier', 'consumptionAbility',
                   'education', 'gender', 'advertiserId', 
                   'campaignId', 'creativeId', 'adCategoryId',
                   'productId', 'productType', 'uid', 'aid']
vector_feature = ['interest1', 'interest2', 'interest5', 'kw1',
                  'kw2', 'topic1', 'topic2', 'os', 'ct', 'marriageStatus']
aid_inter_oh_feature = []  # ['LBS', 'age', 'carrier', 'consumptionAbility', 'education', 'gender']
aid_inter_vec_feature = ['interest1', 'interest2', 'interest5']  # ['interest1', 'interest2', 'interest5', 'kw1', 'kw2', 'topic1', 'topic2']
continous_feature = ['creativeSize']  # + stat_feature
df = df[one_hot_feature + vector_feature + continous_feature]


# In[9]:


ffm_fmt = FFMFormat(vector_feature, one_hot_feature, continous_feature)
user_ffm = ffm_fmt.fit_transform(df)
user_ffm.to_csv(ROOT_PATH + 'data/output/feat/ffm/ffm.csv', index=False)


# In[10]:


train_y = np.array(train.pop('label'))
with open(ROOT_PATH + 'data/output/feat/ffm/ffm.csv') as f_in:
    f_out_train = open(ROOT_PATH + 'data/output/feat/ffm/train_ffm.csv', 'w')
    f_out_test = open(ROOT_PATH + 'data/output/feat/ffm/test_ffm.csv', 'w')
    for (i, line) in enumerate(f_in):
        if i % 100000 == 0:
            log('Iteration: ' + str(i))
        if i < TRAIN_LEN:
            f_out_train.write(str(train_y[i]) + ' ' + line)
        else:
            f_out_test.write(line)
    f_out_train.close()
    f_out_test.close()


# In[ ]:


########################################### xLearn Model ###########################################


# In[11]:


ffm_model = xl.create_ffm()
ffm_model.setTrain(ROOT_PATH + 'data/output/feat/ffm/train_ffm.csv')
ffm_model.setTest(ROOT_PATH + 'data/output/feat/ffm/test_ffm.csv')
ffm_model.setSigmoid()
ffm_model.disableEarlyStop()


# In[ ]:


param = {
    'task': 'binary',
    'lr': 0.01,
    'lambda': 0.001,
    'metric': 'auc',
    'opt': 'ftrl',
    'epoch': 7,  # 5
    'k': 2,
    'alpha': 1.5,
    'beta': 0.01,
    'lambda_1': 0.0,
    'lambda_2': 0.0
}


# In[ ]:


ffm_model.cv(param)


# In[ ]:


ffm_model.fit(param, ROOT_PATH + 'data/output/model/ffm/model.out')


# In[ ]:


ffm_model.predict(ROOT_PATH + 'data/output/model/ffm/model.out', ROOT_PATH + 'data/output/sub/ffm/output.txt')


# In[ ]:


sub = pd.DataFrame()
sub['aid'] = test['aid']
sub['uid'] = test['uid']
sub['score'] = np.loadtxt(ROOT_PATH + 'data/output/sub/ffm/output.txt')
sub.to_csv(ROOT_PATH + 'data/output/sub/ffm/20180516.csv', index=False)

