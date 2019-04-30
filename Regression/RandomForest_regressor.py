#RANDOM FOREST REGRESSOR

#import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#model
from sklearn.ensemble import RandomForestRegressor

#load the data as df using pandas
df_insurance = pd.read_csv('/Datasets/Insurance.csv')

#Basic EDA

df_insurance.info()
df_insurance.describe()
df_insurance.shape #(10, 2)
df_insurance.head()

#Separate the dependent and independent values
X = df_insurance.iloc[:,0:1].values
y = df_insurance.iloc[:,1].values

print(type(X)) #<class 'numpy.ndarray'>
print(type(y)) #<class 'numpy.ndarray'>

print(X.shape) #(10, 1)
print(y.shape) #(10,)

#Fit RandomForest Regressor on X
RF_regressor = RandomForestRegressor(n_estimators=10,random_state=0)
RF_regressor.fit(X,y)

y_pred = RF_regressor.predict(X)
RF_regressor.score(X,y) #0.9704434230386582

#Plot
plt.scatter(X,y,color='red')
plt.plot(X,y_pred,color='blue')
plt.title('Insurance Premium Claim - RF Regressor')
plt.xlabel('Age')
plt.ylabel('Premium Claims')
plt.show()

#Testing the regressor prediction on a unknown value
new_age = 100
pred = RF_regressor.predict(new_age)
print('RF Prediction:',pred) #RF Prediction: [85000.]
