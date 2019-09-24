import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

train = pd.read_csv(r"C:\Users\ABHAY JAMWAL\Desktop\Classes\Projects\Assignment_Data_Science (1)\assignment_train.csv")

'''
outliers=[]
def outlier(data):
   
    threshold=3
    mean_1 = np.mean(data)
    std_1 =np.std(data)
   
   
    for y in data:
        z_score= (y - mean_1)/std_1
        if np.abs(z_score) > threshold:
            outliers.append(y)
    return outliers
outlier(train['3M_all_avg'])

Q1 = train['3M_all_avg'].quantile(0.25)
Q3 = train['3M_all_avg'].quantile(0.75)
IQR = Q3 - Q1
print(IQR)
fence_low  = Q1-1.5*IQR
fence_high = Q3+1.5*IQR

x = train.loc[(train['3M_all_avg'] > fence_low) & (train['3M_all_avg'] < fence_high)]
len(x)
a'''

'''Creating correlation matrix and selecting variables having correlation > 0.9'''
corr_matrix = train.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]
to_drop



train.drop(to_drop,axis =1 ,inplace = True)

train_copy = train

train_copy.iloc[:,0:58].head()


'''Scaling the data and creating final ready to use dataset'''


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(train_copy.iloc[:,0:58])

train_scaled = sc.transform(train_copy.iloc[:,0:58])
train_final = pd.DataFrame(data=train_scaled)
train_final = pd.concat([train_final,train_copy[['business_risk']]],axis=1)

                         '''Splitting the data in 70-30 ratio'''

X = train_final.iloc[:,0:58]
y = train_final['business_risk']
X.head()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
X_train.shape,X_test.shape,y_train.shape,y_test.shape

                    '''Creating object of Randomforest and tuning it'''

rf = RandomForestRegressor()

grid_param = {'n_estimators':[int(x) for x in np.linspace(start=100,stop=830,num=9)],
    'max_depth': [5, 10, 15, 20, 25, 30],
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': [1, 2, 5, 10,50],
 'min_samples_split': [2, 5, 10, 15, 100] }

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = grid_param, 
                             cv = 3, random_state=42, n_jobs = -1)

rf_random.fit(X_train,y_train)

rf = RandomForestRegressor(n_estimators =556,
 min_samples_split= 2,
 min_samples_leaf = 2,
 max_features = 'auto',
 max_depth = 20)

                    '''Fitting the data with selected hyperparameters'''
 
rf.fit(X_train,y_train)
y_pred_rf = rf.predict(X_test)
print('-'*90)
print("Model Score for Training data: {}".format(rf.score(X_train,y_train)))
print("Model Score for Testing data: {}".format(r2_score(y_test,y_pred_rf)))
print("Root Mean Squared Error is {}".format(np.sqrt(mean_squared_error(y_test,y_pred_rf))))

                '''Creating object of XGB and tuning it'''

param_xgb = {'objective': ['reg:linear'],
              'learning_rate' : [.03, 0.05, .07],
              'gamma':[0.5, 1, 1.5, 2, 5],
              'max_depth': [4,3,5, 6, 7],
              'min_child_weight': [1, 5, 10],
              'silent': [1],
              'subsample': [0.6, 0.8, 1.0],
              'colsample_bytree': [0.6, 0.8, 1.0],
              'n_estimators' : [200, 300,400, 500]
                  }

xgb_randomcv = RandomizedSearchCV(xgb,
                        param_xgb,
                        cv = 5,
                        n_jobs = -1,
                        verbose=True)

xgb_randomcv.fit(X_train,y_train)
xgb_randomcv.best_params_


xgb = XGBRegressor(subsample = 0.8,
 silent=1,
 objective = 'reg:linear',
 n_estimators = 400,
 min_child_weight =  10,
 max_depth = 7,
 learning_rate = 0.07,
 gamma= 1.5,
 colsample_bytree = 1.0)

                            'Fitting the data with selected hyperparameters'''

xgb.fit(X_train,y_train)
y_pred_xgb = xgb.predict(X_test)
print('-'*90)
print("Model Score for Training data: {}".format(xgb.score(X_train,y_train)))
print("Model Score for Testing data: {}".format(r2_score(y_test,y_pred_xgb)))
print("Root Mean Squared Error is {}".format(np.sqrt(mean_squared_error(y_test,y_pred_xgb))))


                    '''Doing all operations on test data'''

test = pd.read_csv(r"C:\Users\ABHAY JAMWAL\Desktop\Classes\Projects\Assignment_Data_Science (1)\assignment_test.csv")

corr_matrix_test = test.corr().abs()
upper_test = corr_matrix_test.where(np.triu(np.ones(corr_matrix_test.shape), k=1).astype(np.bool))
to_drop_test = [column for column in upper_test.columns if any(upper_test[column] > 0.90)]

test.drop(to_drop,axis =1 ,inplace = True)

test.drop('agent_id',axis= 1,inplace= True)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(test.iloc[:,0:58])

test_scaled = sc.transform(test.iloc[:,0:58])

test_final = pd.DataFrame(data=test_scaled)
test_final = pd.concat([test_final])

                '''Predicting the test data with XGB and exporting to csv'''

pred_test = xgb.predict(test_final.iloc[:,0:58])
pred_test
test_final['business_coeff'] = pred_test
test_final.head()

test_final.to_csv(r"C:\Users\ABHAY JAMWAL\Desktop\Classes\Projects\Assignment_Data_Science (1)\submission_1.csv")

