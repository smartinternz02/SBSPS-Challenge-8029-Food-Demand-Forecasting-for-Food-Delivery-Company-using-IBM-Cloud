#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle
from sklearn.preprocessing import LabelEncoder


# In[2]:


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# ### Understanding the Dataset

# In[3]:


train.head()


# In[4]:


test.head()


# In[5]:


train.info()


# In[6]:


train.describe()


# In[7]:


test.info()


# We can understand that we have to predict the values of column 'num_orders' for the test data. 

# #### Checking for Null Values

# In[8]:


train.isnull().sum()


# In[9]:


test.isnull().sum()


# This shows that the train and test datasets are not having the null values

# #### Merging the mean_info.csv and center_info.csv datasets
# 
# This step can help us to select other necessary features for the prediction of the number of orders and thus knowing the demand

# In[10]:


info_meal = pd.read_csv("meal_info.csv")
info_center = pd.read_csv("fulfilment_center_info.csv")


# In[11]:


info_meal.head()


# In[12]:


info_center.head()


# In[13]:


train_f  = pd.merge(train,info_meal,on = 'meal_id',how = 'outer')
train_f = pd.merge(train_f,info_center, on = 'center_id', how = 'outer')


# In[14]:


train_f.head()


# #### Droping columns
# 
# This steps removes the features from the dataset which do not contribute in prediction of number of orders.
# We can observe that mean id and center id don't affect the number of orders. Hence we will drop those columns

# In[15]:


train_f = train_f.drop(['center_id','meal_id'],axis = 'columns')
train_f.head()


# Reordering Columns

# In[16]:


cols = train_f.columns.to_list()
cols


# In[17]:


cols = cols[:2] + cols[9:] + cols[7:9] + cols[2:7]
print(cols)


# In[18]:


train_f = train_f[cols]
train_f.dtypes


# After meging the datasets, we can observe that we also have object type dataset along with numerical data. So we need to convert the object type data to numerical data for better analysis using label encoding

# In[19]:


le_cntr_typ = LabelEncoder()
le_ctgry = LabelEncoder()
le_cusine = LabelEncoder()


# In[20]:


train_f['center_type'] = le_cntr_typ.fit_transform(train_f['center_type'])
train_f['category'] = le_ctgry.fit_transform(train_f['category'])
train_f['cuisine'] = le_cusine.fit_transform(train_f['cuisine'])


# In[21]:


train_f.head()


# In[22]:


train_f.shape


# Visualising data: 

# In[25]:


plt.style.use('fivethirtyeight')
plt.figure(figsize = (12,7))
sns.distplot(train_f.num_orders,bins = 25)
plt.xlabel('num_orders')
plt.ylabel('number of buyers')
plt.title('num_order distribution')


# In[26]:


bin_width = max(train['num_orders'])/25
bin_width


# In[27]:


n = train['num_orders'] < 972
print('no of customers in the bin with orders less than 972 (bin with max freq from the above graph): ' + str(n.sum()))
print('Total number of customers: ' + str(len(train)))


# Majority of the buyers are found having the number of orders between 0 and 972 

# In[28]:


train_f2 = train_f.drop(['id'],axis = 'columns')
corr = train_f2.corr(method = 'pearson')
columns = corr.nlargest(8,'num_orders').index
columns


# In[29]:


corr_map = np.corrcoef(train_f2[columns].values.T)
sns.set(font_scale = 1)
heat_map = sns.heatmap(corr_map,cbar = True, annot = True, square = True, fmt = '.2f', yticklabels = columns.values, xticklabels = columns.values)


# We can observe that, num_orders is more likely to get affected by featuring homepage and having an emailer for promotions. In the top 7 columns which affect the number of orders positively are: homepage_featured, emailer_for_promotion, op_area, cuisine, city_code, region_code and category (i.e. type of food ordered, eg: beverage)

# ### Getting Ready for applying algorithms of ML

# In[30]:


# num_orders - > dependent variable
# other top 7 rows having the max correlation with the num_orders -> independent variables

features = columns.drop(['num_orders'])
train_f3 = train_f[features]
X = train_f3.values
y = train_f['num_orders'].values


# In[31]:


train_f3.head()


# In[32]:


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X,y,test_size = 0.25)


# In[33]:


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn import metrics


# In[34]:


XG = XGBRegressor()
XG.fit(X_train,y_train)
y_pred = XG.predict(X_val)
y_pred[y_pred<0] = 0
print('RMSLE: ',100*np.sqrt(metrics.mean_squared_log_error(y_val,y_pred)))


# In[35]:


LR = LinearRegression()
LR.fit(X_train,y_train)
y_pred = LR.predict(X_val)
y_pred[y_pred<0] = 0
print('RMSLE: ',100*np.sqrt(metrics.mean_squared_log_error(y_val,y_pred)))


# In[36]:


L = Lasso()
L.fit(X_train,y_train)
y_pred = L.predict(X_val)
y_pred[y_pred<0] = 0
print('RMSLE: ',100*np.sqrt(metrics.mean_squared_log_error(y_val,y_pred)))


# In[37]:


EN = ElasticNet()
EN.fit(X_train,y_train)
y_pred = EN.predict(X_val)
y_pred[y_pred<0] = 0
print('RMSLE: ',100*np.sqrt(metrics.mean_squared_log_error(y_val,y_pred)))


# In[38]:


DT = DecisionTreeRegressor()
DT.fit(X_train,y_train)
y_pred = DT.predict(X_val)
y_pred[y_pred<0] = 0
print('RMSLE: ',100*np.sqrt(metrics.mean_squared_log_error(y_val,y_pred)))


# In[39]:


KNN = KNeighborsRegressor()
KNN.fit(X_train,y_train)
y_pred = KNN.predict(X_val)
y_pred[y_pred<0] = 0
print('RMSLE: ',100*np.sqrt(metrics.mean_squared_log_error(y_val,y_pred)))


# In[40]:


GB = GradientBoostingRegressor()
GB.fit(X_train,y_train)
y_pred = GB.predict(X_val)
y_pred[y_pred<0] = 0
print('RMSLE: ',100*np.sqrt(metrics.mean_squared_log_error(y_val,y_pred)))


# By observation, Decission tree regressor is giving the least Root mean square log error. Hence, we will use decision tree as our model to predict the values

# In[41]:


pickle.dump(DT,open('fdemand.pkl','wb'))


# ### Prediction
# Preparing the test dataset

# In[42]:


test_f  = pd.merge(test,info_meal,on = 'meal_id',how = 'outer')
test_f = pd.merge(test_f,info_center, on = 'center_id', how = 'outer')
test_f.head()


# In[43]:


test_f = test_f.drop(['center_id','meal_id'],axis = 'columns')
test_f.head()


# In[44]:


t_cols = test_f.columns.to_list()
print(t_cols)


# In[45]:


t_cols = t_cols[:2] + t_cols[8:] + t_cols[6:8] + t_cols[2:6]
print(t_cols)


# In[46]:


test_f = test_f[t_cols]


# In[47]:


le_cntr_typ = LabelEncoder()
le_ctgry = LabelEncoder()
le_cusine = LabelEncoder()

test_f['center_type'] = le_cntr_typ.fit_transform(test_f['center_type'])
test_f['category'] = le_ctgry.fit_transform(test_f['category'])
test_f['cuisine'] = le_cusine.fit_transform(test_f['cuisine'])


# In[48]:


test_f.head()


# In[49]:


X_test = test_f[features].values


# In[50]:


pred = DT.predict(X_test)
pred[pred<0] = 0


# In[51]:


submit = pd.DataFrame({ 'id' : test_f['id'],'num_orders' : pred })


# In[52]:


submit.head()


# In[53]:


submit.describe()


# In[54]:


submit.to_excel('submission.xlsx')

