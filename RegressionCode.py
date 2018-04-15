import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt

data=pd.read_csv('luxv.csv')
names = ['Voltage OF PD', 'LUX of sensor']
data.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)

df=pd.DataFrame(data,columns=names)
features = pd.get_dummies(df)
labels = np.array(features['LUX of sensor'])
features= features.drop('LUX of sensor', axis = 1)
feature_list = list(features.columns)
features = np.array(features)

## split data into training and labels
train_features, test_features, train_labels,test_labels = train_test_split(features, labels,test_size = 0.25, random_state = 42)


## predicion via randomforset regressor
rf = RandomForestRegressor(n_estimators = 1000, random_state = 40)
rf.fit(train_features, train_labels)
predictedrf= rf.predict(test_features)
accr=rf.score(test_features,test_labels)
print('randomforst accuracy=',accr)

## predicion via Linear regressor
LRegg=linear_model.LinearRegression()
LRegg.fit(train_features, train_labels)
predictedLr= LRegg.predict(test_features)
accr2=LRegg.score(test_features,test_labels)
print('linear regression accuracy=',accr2)

## predicion via K nearest neighbors regressor
kn=KNeighborsRegressor(n_neighbors=3)
kn.fit(train_features, train_labels)
predictedkn= kn.predict(test_features)
accr3=kn.score(test_features,test_labels)
print('knn accuracy=',accr3)


## predicion via GB regressor
gb= GradientBoostingRegressor(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
gb.fit(train_features,train_labels)
predictedgb=gb.predict(test_features)
accr5=gb.score(test_features,test_labels)
print('GB accuacy=',accr5)

###Output
randomforst accuracy= 0.978791136646
linear regression accuracy= 0.933613443461
knn accuracy= 0.984196151073
GradientBoosting accuacy= 0.981459203765




