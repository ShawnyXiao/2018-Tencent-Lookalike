
# coding: utf-8

# In[1]:


from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import cPickle
import gc
import time
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


# In[ ]:


########################################### Read data ###########################################


# In[5]:


df = cPickle.load(open(ROOT_PATH + 'data/output/lgb/feat/train+test1+test2/all(basic).p', 'rb'))

train = df.iloc[:train_len, :]
test = df.iloc[train_len:, :]

del df, test


# In[3]:


test1 = pd.read_csv(ROOT_PATH + 'data/input/final/test1.csv', header=0)
test2 = pd.read_csv(ROOT_PATH + 'data/input/final/test2.csv', header=0)


# In[ ]:


########################################### Analyze ###########################################


# In[9]:


log(len(train[train[target] == 1]))
log(len(train[train[target] == 0]))
log(len(train[train[target] == 1]) / len(train[train[target] == 0]))  # 1:20


# In[12]:


log(len(np.unique(train.aid)))
log(len(np.unique(train.uid)))


# In[21]:


data = train.groupby('aid')[target].mean()

plt.figure(figsize=(15, 6))
sns.barplot(data.index, data.values)
plt.xlabel('aid')
plt.ylabel('cvr')
plt.show()


# In[22]:


data = train.groupby('aid')[target].count()

plt.figure(figsize=(15, 6))
sns.barplot(data.index, data.values)
plt.xlabel('aid')
plt.ylabel('cvr')
plt.show()


# In[43]:


data = train.groupby('aid')[target].count().reset_index()
data.columns = ['aid', 'count']
data = data.groupby('count').aid.count()

plt.figure(figsize=(15, 6))
sns.barplot(data.index, data.values)
plt.xlabel('count')
plt.ylabel('count of aid')
plt.show()
log(data.index[0])
log(data.index[-1])


# In[9]:


log(len(test1))
log(len(test2))
log(len(test1.append(test2).reset_index(drop=True).drop_duplicates()))


# In[6]:


['LBS', 'age', 'carrier', 'consumptionAbility', 'education', 'gender', 'house', 'os', 'ct']

data = train.groupby(['aid', 'LBS'])[target].mean().reset_index()
data = data.pivot('aid', 'LBS', target)

plt.figure(figsize=(15, 6))
sns.heatmap(data)
plt.show()


# In[7]:


data = train.groupby(['aid', 'age'])[target].mean().reset_index()
data = data.pivot('aid', 'age', target)

plt.figure(figsize=(15, 6))
sns.heatmap(data)
plt.show()


# In[8]:


data = train.groupby(['aid', 'carrier'])[target].mean().reset_index()
data = data.pivot('aid', 'carrier', target)

plt.figure(figsize=(15, 6))
sns.heatmap(data)
plt.show()


# In[9]:


data = train.groupby(['aid', 'consumptionAbility'])[target].mean().reset_index()
data = data.pivot('aid', 'consumptionAbility', target)

plt.figure(figsize=(15, 6))
sns.heatmap(data)
plt.show()


# In[10]:


data = train.groupby(['aid', 'education'])[target].mean().reset_index()
data = data.pivot('aid', 'education', target)

plt.figure(figsize=(15, 6))
sns.heatmap(data)
plt.show()


# In[11]:


data = train.groupby(['aid', 'gender'])[target].mean().reset_index()
data = data.pivot('aid', 'gender', target)

plt.figure(figsize=(15, 6))
sns.heatmap(data)
plt.show()


# In[12]:


data = train.groupby(['aid', 'house'])[target].mean().reset_index()
data = data.pivot('aid', 'house', target)

plt.figure(figsize=(15, 6))
sns.heatmap(data)
plt.show()


# In[13]:


data = train.groupby(['aid', 'os'])[target].mean().reset_index()
data = data.pivot('aid', 'os', target)

plt.figure(figsize=(15, 6))
sns.heatmap(data)
plt.show()


# In[14]:


data = train.groupby(['aid', 'ct'])[target].mean().reset_index()
data = data.pivot('aid', 'ct', target)

plt.figure(figsize=(15, 6))
sns.heatmap(data)
plt.show()


# In[19]:


['LBS', 'age', 'carrier', 'consumptionAbility', 'education', 'gender', 'house', 'os', 'ct']
['advertiserId', 'campaignId', 'creativeId', 'adCategoryId', 'productId', 'productType']

log(len(np.unique(train.LBS.values)))
log(len(np.unique(train.age.values)))
log(len(np.unique(train.carrier.values)))
log(len(np.unique(train.consumptionAbility.values)))
log(len(np.unique(train.education.values)))
log(len(np.unique(train.gender.values)))
log(len(np.unique(train.house.values)))
log(len(np.unique(train.os.values)))
log(len(np.unique(train.ct.values)))
log('')
log(len(np.unique(train.advertiserId.values)))
log(len(np.unique(train.campaignId.values)))
log(len(np.unique(train.creativeId.values)))
log(len(np.unique(train.adCategoryId.values)))
log(len(np.unique(train.productId.values)))
log(len(np.unique(train.productType.values)))


# In[36]:


data = train.groupby(['adCategoryId', 'age'])[target].mean().reset_index()
data = data.pivot('adCategoryId', 'age', target)

plt.figure(figsize=(15, 6))
sns.heatmap(data)
plt.show()


# In[37]:


data = train.groupby(['adCategoryId', 'carrier'])[target].mean().reset_index()
data = data.pivot('adCategoryId', 'carrier', target)

plt.figure(figsize=(15, 6))
sns.heatmap(data)
plt.show()


# In[38]:


data = train.groupby(['adCategoryId', 'consumptionAbility'])[target].mean().reset_index()
data = data.pivot('adCategoryId', 'consumptionAbility', target)

plt.figure(figsize=(15, 6))
sns.heatmap(data)
plt.show()


# In[39]:


data = train.groupby(['adCategoryId', 'education'])[target].mean().reset_index()
data = data.pivot('adCategoryId', 'education', target)

plt.figure(figsize=(15, 6))
sns.heatmap(data)
plt.show()


# In[40]:


data = train.groupby(['adCategoryId', 'gender'])[target].mean().reset_index()
data = data.pivot('adCategoryId', 'gender', target)

plt.figure(figsize=(15, 6))
sns.heatmap(data)
plt.show()


# In[41]:


data = train.groupby(['adCategoryId', 'house'])[target].mean().reset_index()
data = data.pivot('adCategoryId', 'house', target)

plt.figure(figsize=(15, 6))
sns.heatmap(data)
plt.show()


# In[42]:


data = train.groupby(['adCategoryId', 'os'])[target].mean().reset_index()
data = data.pivot('adCategoryId', 'os', target)

plt.figure(figsize=(15, 6))
sns.heatmap(data)
plt.show()


# In[43]:


data = train.groupby(['adCategoryId', 'ct'])[target].mean().reset_index()
data = data.pivot('adCategoryId', 'ct', target)

plt.figure(figsize=(15, 6))
sns.heatmap(data)
plt.show()


# In[ ]:


['LBS', 'age', 'carrier', 'consumptionAbility', 'education', 'gender', 'house', 'os', 'ct']
['advertiserId', 'campaignId', 'creativeId', 'adCategoryId', 'productId', 'productType']


# In[45]:


data = train.groupby(['consumptionAbility', 'advertiserId'])[target].mean().reset_index()
data = data.pivot('consumptionAbility', 'advertiserId', target)

plt.figure(figsize=(15, 6))
sns.heatmap(data)
plt.show()


# In[46]:


data = train.groupby(['consumptionAbility', 'campaignId'])[target].mean().reset_index()
data = data.pivot('consumptionAbility', 'campaignId', target)

plt.figure(figsize=(15, 6))
sns.heatmap(data)
plt.show()


# In[47]:


data = train.groupby(['consumptionAbility', 'creativeId'])[target].mean().reset_index()
data = data.pivot('consumptionAbility', 'creativeId', target)

plt.figure(figsize=(15, 6))
sns.heatmap(data)
plt.show()


# In[48]:


data = train.groupby(['consumptionAbility', 'adCategoryId'])[target].mean().reset_index()
data = data.pivot('consumptionAbility', 'adCategoryId', target)

plt.figure(figsize=(15, 6))
sns.heatmap(data)
plt.show()


# In[49]:


data = train.groupby(['consumptionAbility', 'productId'])[target].mean().reset_index()
data = data.pivot('consumptionAbility', 'productId', target)

plt.figure(figsize=(15, 6))
sns.heatmap(data)
plt.show()


# In[50]:


data = train.groupby(['consumptionAbility', 'productType'])[target].mean().reset_index()
data = data.pivot('consumptionAbility', 'productType', target)

plt.figure(figsize=(15, 6))
sns.heatmap(data)
plt.show()


# In[ ]:


['LBS', 'age', 'carrier', 'consumptionAbility', 'education', 'gender', 'house', 'os', 'ct']
['advertiserId', 'campaignId', 'creativeId', 'adCategoryId', 'productId', 'productType']


# In[51]:


data = train.groupby(['education', 'advertiserId'])[target].mean().reset_index()
data = data.pivot('education', 'advertiserId', target)

plt.figure(figsize=(15, 6))
sns.heatmap(data)
plt.show()


# In[52]:


data = train.groupby(['education', 'campaignId'])[target].mean().reset_index()
data = data.pivot('education', 'campaignId', target)

plt.figure(figsize=(15, 6))
sns.heatmap(data)
plt.show()


# In[53]:


data = train.groupby(['education', 'creativeId'])[target].mean().reset_index()
data = data.pivot('education', 'creativeId', target)

plt.figure(figsize=(15, 6))
sns.heatmap(data)
plt.show()


# In[54]:


data = train.groupby(['education', 'adCategoryId'])[target].mean().reset_index()
data = data.pivot('education', 'adCategoryId', target)

plt.figure(figsize=(15, 6))
sns.heatmap(data)
plt.show()


# In[55]:


data = train.groupby(['education', 'productId'])[target].mean().reset_index()
data = data.pivot('education', 'productId', target)

plt.figure(figsize=(15, 6))
sns.heatmap(data)
plt.show()


# In[56]:


data = train.groupby(['education', 'productType'])[target].mean().reset_index()
data = data.pivot('education', 'productType', target)

plt.figure(figsize=(15, 6))
sns.heatmap(data)
plt.show()


# In[ ]:


['LBS', 'age', 'carrier', 'consumptionAbility', 'education', 'gender', 'house', 'os', 'ct']
['advertiserId', 'campaignId', 'creativeId', 'adCategoryId', 'productId', 'productType']


# In[57]:


data = train.groupby(['age', 'advertiserId'])[target].mean().reset_index()
data = data.pivot('age', 'advertiserId', target)

plt.figure(figsize=(15, 6))
sns.heatmap(data)
plt.show()


# In[58]:


data = train.groupby(['age', 'campaignId'])[target].mean().reset_index()
data = data.pivot('age', 'campaignId', target)

plt.figure(figsize=(15, 6))
sns.heatmap(data)
plt.show()


# In[59]:


data = train.groupby(['age', 'creativeId'])[target].mean().reset_index()
data = data.pivot('age', 'creativeId', target)

plt.figure(figsize=(15, 6))
sns.heatmap(data)
plt.show()


# In[60]:


data = train.groupby(['age', 'adCategoryId'])[target].mean().reset_index()
data = data.pivot('age', 'adCategoryId', target)

plt.figure(figsize=(15, 6))
sns.heatmap(data)
plt.show()


# In[61]:


data = train.groupby(['age', 'productId'])[target].mean().reset_index()
data = data.pivot('age', 'productId', target)

plt.figure(figsize=(15, 6))
sns.heatmap(data)
plt.show()


# In[62]:


data = train.groupby(['age', 'productType'])[target].mean().reset_index()
data = data.pivot('age', 'productType', target)

plt.figure(figsize=(15, 6))
sns.heatmap(data)
plt.show()

