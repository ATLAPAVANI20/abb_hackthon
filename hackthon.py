import pandas as pd
import numpy as np

train = pd.read_csv(r"C:\Users\Anil\OneDrive\Desktop\hackthon\sample_submission_8RXa3c6.csv")
test = pd.read_csv(r"C:\Users\Anil\OneDrive\Desktop\hackthon\train_v9rqX0R.csv")
ss = pd.read_csv(r"C:\Users\Anil\OneDrive\Desktop\hackthon\sample_submission_8RXa3c6.csv")

train.shape , test.shape , ss.shape
pd.set_option('display.max_columns', 50)
train.head()
# Data Exploration, EDA
train.info()
for col in train.iloc[:,0:21].columns:
    print(col,':',train[col].nunique(),':',train[col].isna().sum())
    #target_mean(train.col,'click_rate_log')
### Target Analysis
import seaborn  
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(20, 3))
seaborn.set(style = 'whitegrid')
seaborn.violinplot(x ="Item_Type", y ="Item_Outlet_Sales",style ="event",data = train)

fig, ax = plt.subplots(figsize=(12, 4))
seaborn.set(style = 'whitegrid')
seaborn.violinplot(x ="Outlet_Identifier", y ="Item_Outlet_Sales",style ="event",data = train)
fig, ax = plt.subplots(figsize=(12, 4))
seaborn.set(style = 'whitegrid')
seaborn.violinplot(x ="Outlet_Size", y ="Item_Outlet_Sales",style ="event",data = train)
fig, ax = plt.subplots(figsize=(12, 4))
seaborn.set(style = 'whitegrid')
seaborn.scatterplot(x ="Item_MRP", y ="Item_Outlet_Sales",data = train)
fig, ax = plt.subplots(figsize=(12, 4))
seaborn.set(style = 'whitegrid')
seaborn.scatterplot(x ="Item_Visibility", y ="Item_Outlet_Sales",data = train)
fig, ax = plt.subplots(figsize=(12, 4))
seaborn.set(style = 'whitegrid')
seaborn.scatterplot(x ="Item_Weight", y ="Item_Outlet_Sales",data = train)
train["Item_Weight"] = train["Item_Weight"].fillna(train.Item_Weight.mean())
test["Item_Weight"] = test["Item_Weight"].fillna(test.Item_Weight.mean())

# Data Processsing and Feature Engineering
train['Item_Fat_Content']=train['Item_Fat_Content'].replace({ 'Regular':1, 'reg':1,'Low Fat':0,'low fat':0,'LF':0})
train['Item_Visibility'] =  np.where(train['Item_Visibility'] == 0,"NaN",train['Item_Visibility']).astype(float)
train['Item_Identifier'] = train['Item_Identifier'].str.slice(0,2)
train['running'] = 2013 - train['Outlet_Establishment_Year']
train['price/wt'] = train['Item_MRP'] /train['Item_Weight'] 
train['Outlet_Size'] =  np.where(train['Outlet_Identifier'] == 'OUT010',"Small",train['Outlet_Size'])
train['Outlet_Size'] =  np.where(train['Outlet_Identifier'] == 'OUT017',"Medium",train['Outlet_Size'])
train['Outlet_Size'] =  np.where(train['Outlet_Identifier'] == 'OUT045',"Medium",train['Outlet_Size'])
train['Item_MRP2'] =  np.where(train['Item_MRP'] <69,"A",
                              np.where(train['Item_MRP'] <136,"B",
                                       np.where(train['Item_MRP'] <203,"C","D")))
#train['Item_Visibility2'] =  np.where(train['Item_Visibility'] < 0.19,1,0)



test['Item_Fat_Content']=test['Item_Fat_Content'].replace({ 'Regular':1, 'reg':1,'Low Fat':0,'low fat':0,'LF':0})
test['Item_Visibility'] =  np.where(test['Item_Visibility'] == 0,"NaN",test['Item_Visibility']).astype(float)
test['Item_Identifier'] = test['Item_Identifier'].str.slice(0,2)
test['running'] = 2013 - test['Outlet_Establishment_Year']
test['price/wt'] = test['Item_MRP'] /test['Item_Weight']
test['Outlet_Size'] =  np.where(test['Outlet_Identifier'] == 'OUT010',"Small",test['Outlet_Size'])
test['Outlet_Size'] =  np.where(test['Outlet_Identifier'] == 'OUT017',"Medium",test['Outlet_Size'])
test['Outlet_Size'] =  np.where(test['Outlet_Identifier'] == 'OUT045',"Medium",test['Outlet_Size'])
test['Item_MRP2'] =  np.where(test['Item_MRP'] <69,"A",
                              np.where(test['Item_MRP'] <136,"B",
                                       np.where(test['Item_MRP'] <203,"C","D")))   
#test['Item_Visibility2'] =  np.where(test['Item_Visibility'] < 0.19,1,0)
# Check co-relation with target for numeric columns
import seaborn as sns
import numpy as np
corr=train[['Item_Weight','Item_Visibility','Item_MRP','running','price/wt','Item_Outlet_Sales']].corr()
mask=np.triu(np.ones_like(corr))
sns.heatmap(corr,annot=True,mask=mask,cbar=False)
y_train = train['Item_Outlet_Sales']
x_train = train.drop(['Item_Outlet_Sales','Outlet_Establishment_Year'],axis=1)
x_train.head()
from catboost import CatBoostRegressor, Pool
categorical_features =  np.where(x_train.dtypes == object )[0]

def objective(trial,data=x_train,target=y_train):
    
    train_x, test_x, train_y, test_y = train_test_split(x_train, y_train, test_size=0.15,random_state=42)
    param = {
        'loss_function': 'RMSE',
        #'task_type': 'GPU',
        'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-3, 10.0),
        'max_bin': trial.suggest_int('max_bin', 200, 400),
        #'rsm': trial.suggest_uniform('rsm', 0.3, 1.0),
        'subsample': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
        'learning_rate': trial.suggest_uniform('learning_rate', 0.006, 0.018),
        'n_estimators':  1000,
        'max_depth': trial.suggest_categorical('max_depth', [5,7,9,11,13,15]),
        'random_state': trial.suggest_categorical('random_state', [2020]),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 300),
    }
    model = CatBoostRegressor(**param,cat_features=categorical_features)  
    
    model.fit(train_x,train_y,eval_set=[(test_x,test_y)],early_stopping_rounds=200,verbose=False)
    
    preds = model.predict(test_x)
    
    rmse = mean_squared_error(test_y, preds,squared=False)
    
    return rmse
Best_trial = {'l2_leaf_reg': 0.001061926310,'max_bin': 322,
 'learning_rate': 0.01081467174,'max_depth': 5,'random_state': 2020,'min_data_in_leaf': 163,
              'loss_function': 'RMSE','n_estimators':  1000}
from catboost import CatBoostRegressor, Pool
categorical_features =  np.where(x_train.dtypes == object )[0]

model = CatBoostRegressor(**Best_trial,cat_features=categorical_features)
model.fit(x_train, y_train)

test_pred = model.predict(test[x_train.columns])
test_pred[test_pred<33]=33
ss['Item_Outlet_Sales'] = test_pred
ss.to_csv('bigmart.csv',index=False)
ss.head()