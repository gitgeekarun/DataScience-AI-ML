#DECISION TREE REGRESSOR
#import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#model
from sklearn.tree import DecisionTreeRegressor

#metrics
from sklearn.metrics import r2_score,mean_squared_error

#load the data as df using pandas
df_insurance = pd.read_csv('/Datasets/Insurance.csv')

#Basic EDA
df_price.info()
df_price.describe()
df_price.shape #(10, 2)
df_price.head()

#Separate the dependent and independent values
X = df_insurance.iloc[:,0:1].values
y = df_insurance.iloc[:,1].values

print(type(X)) #<class 'numpy.ndarray'>
print(type(y)) #<class 'numpy.ndarray'>

print(X.shape) #(10, 1)
print(y.shape) #(10,)

#Fit DT Regressor on X
DT_regressor = DecisionTreeRegressor(random_state=0)
DT_regressor.fit(X,y)

y_pred = DT_regressor.predict(X)

#plot
plt.scatter(X,y,color='red')
plt.plot(X,y_pred,color='blue')
plt.title('Insurance Premium Claim - DT Regressor')
plt.xlabel('Age')
plt.ylabel('Premium Claims')
plt.show()

#Testing the regressor prediction on a unknown value
new_age = 67
pred = DT_regressor.predict(new_age)
print(pred) #[50000.]

