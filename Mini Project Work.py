#!/usr/bin/env python
# coding: utf-8

# # Price Predicting module for Cars

# In[1]:


import pandas as pd


# In[2]:


car_pricing = pd.read_excel("data.xlsx")


# ## data first glance

# In[3]:


car_pricing.head()


# In[4]:


car_pricing.info()


# In[5]:


car_pricing['Year']


# In[6]:


car_pricing['Year'].value_counts()


# In[7]:


car_pricing.describe()


# In[8]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


import matplotlib.pyplot as plt


# In[10]:


car_pricing.hist(bins=50, figsize=(20,15))


# ### Data set should not repeat

# In[11]:


import numpy as np
def split_train_test(data, test_ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    print(shuffled)
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


# In[12]:


#train_set, test_set = split_train_test(car_pricing, 0.2)


# In[13]:


#print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}\n")


# ## train test splitting

# In[14]:


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(car_pricing, test_size=0.2, random_state=42)
print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}\n")


# In[15]:


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(car_pricing, car_pricing['Year']):
    strat_train_set = car_pricing.loc[train_index]
    strat_test_set = car_pricing.loc[test_index]


# In[16]:


strat_train_set['Year'].value_counts()


# In[17]:


car_pricing = strat_train_set.copy()


# ## Looking for Correlations

# In[18]:


corr_matrix = car_pricing.corr('pearson')
corr_matrix['Price'].sort_values (ascending=False)


# ### understanding data relations bwtween each other

# In[19]:


from pandas.plotting import scatter_matrix
attributes = ["Price", "Year", "Mileage", "Kilometers_Driven", "Seats", "Owner_Type", "Engine"]
scatter_matrix(car_pricing[attributes], figsize = (12,8))


# In[20]:


car_pricing.plot(kind="scatter", x="Year", y="Price", alpha=0.8)


# In[21]:


car_pricing = strat_train_set.drop("Price", axis=1)
car_pricing_labels = strat_train_set["Price"].copy()


# ## filling missing attributes

# ### switching null values with median

# In[22]:


median = car_pricing["Seats"].median()


# In[23]:


car_pricing["Seats"].fillna(median)


# In[24]:


car_pricing.shape


# #### Data type Conversion

# In[25]:


car_pricing['Year'] = car_pricing['Year'].astype(float)
car_pricing['Kilometers_Driven'] = car_pricing['Kilometers_Driven'].astype(float)
car_pricing['Owner_Type'] = car_pricing['Owner_Type'].astype(float)
car_pricing['Mileage'] = car_pricing['Mileage'].astype(float)
car_pricing['Engine'] = car_pricing['Engine'].astype(float)
car_pricing['Seats'] = car_pricing['Seats'].astype(float)
#car_pricing['Price'] = car_pricing['Price'].astype(float)


# In[26]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
imputer.fit(car_pricing)


# In[27]:


imputer.statistics_


# In[28]:


X = imputer.transform(car_pricing)


# In[29]:


car_pricing_tr = pd.DataFrame(X, columns=car_pricing.columns)


# In[30]:


car_pricing_tr.describe()


# ## Scikit - Learn Design

# ### Creating a Pipeline

# In[31]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler()),
])


# In[32]:


car_pricing_num_tr = my_pipeline.fit_transform(car_pricing)


# In[33]:


car_pricing_tr.shape


# ## Selecting a Desired Model

# ### Linear Regression

# In[34]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(car_pricing_num_tr, car_pricing_labels)


# In[35]:


some_data = car_pricing.iloc[:5]


# In[36]:


some_labels = car_pricing.iloc[5:]


# In[37]:


prepared_data = my_pipeline.transform(some_data)


# In[38]:


model.predict(prepared_data)


# In[39]:


some_labels


# #### Evaluating the Model

# In[40]:


from sklearn.metrics import mean_squared_error
car_pricing_predictions = model.predict(car_pricing_num_tr)
lin_mse = mean_squared_error(car_pricing_labels, car_pricing_predictions)
lin_rmse = np.sqrt(lin_mse)


# In[41]:


lin_rmse


# ### Decision Tree Regression

# In[42]:


from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(car_pricing_num_tr, car_pricing_labels)


# In[43]:


some_data = car_pricing.iloc[:5]


# In[44]:


some_labels = car_pricing.iloc[5:]


# In[45]:


prepared_data = my_pipeline.transform(some_data)


# In[46]:


model.predict(prepared_data)


# In[47]:


some_labels


# #### Evaluating the Model

# In[48]:


from sklearn.metrics import mean_squared_error
car_pricing_predictions = model.predict(car_pricing_num_tr)
mse = mean_squared_error(car_pricing_labels, car_pricing_predictions)
rmse = np.sqrt(mse)


# In[49]:


rmse


# ### Random Forest Regression

# In[50]:


from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(car_pricing_num_tr, car_pricing_labels)


# In[51]:


some_data = car_pricing.iloc[:5]


# In[52]:


some_labels = car_pricing.iloc[5:]


# In[53]:


prepared_data = my_pipeline.transform(some_data)


# In[54]:


model.predict(prepared_data)


# In[55]:


some_labels


# In[56]:


from sklearn.metrics import mean_squared_error
car_pricing_predictions = model.predict(car_pricing_num_tr)
mse = mean_squared_error(car_pricing_labels, car_pricing_predictions)
rf_mse = np.sqrt(mse)


# In[57]:


rf_mse


# ## Using Cross Validation

# ### for Linear Regression

# In[58]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, car_pricing_num_tr, car_pricing_labels, scoring="neg_mean_squared_error")
lin_rmse_scores = np.sqrt(-scores)


# In[59]:


lin_rmse_scores


# In[60]:


def print_scores(scores):
    print("Scores:",scores)
    print("Mean: ",scores.mean())
    print("Standard deviation:",scores.std())


# In[61]:


print_scores(lin_rmse_scores)


# ### for Decision Tree Regressor

# In[62]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, car_pricing_num_tr, car_pricing_labels, scoring="neg_mean_squared_error")
rmse_scores = np.sqrt(-scores)


# In[63]:


rmse_scores


# In[64]:


def print_scores(scores):
    print("Scores:",scores)
    print("Mean: ",scores.mean())
    print("Standard deviation:",scores.std())


# In[65]:


print_scores(rmse_scores)


# ### for Random Forest Regressor

# In[66]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, car_pricing_num_tr, car_pricing_labels, scoring="neg_mean_squared_error")
rf_mse_scores = np.sqrt(-scores)


# In[67]:


rf_mse_scores


# In[68]:


def print_scores(scores):
    print("Scores:",scores)
    print("Mean: ",scores.mean())
    print("Standard deviation:",scores.std())


# In[69]:


print_scores(rf_mse_scores)


# In[70]:


from joblib import dump, load
dump(model, 'Fly car 69.joblib')


# ## Testing the model on Test Data

# In[71]:


X_test = strat_test_set.drop("Price", axis=1)
Y_test = strat_test_set["Price"].copy()
X_test_prepared = my_pipeline.transform(X_test)
final_predictions = model.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test, final_predictions)
final_rf_mse = np.sqrt(final_mse)
#print(final_predictions, list(Y_test))


# In[72]:


final_rf_mse


# In[73]:


#prepared_data[0]


# In[ ]:




