#import all required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#models
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

#Read the data as dataframe using pandas
df_insurance = pd.read_csv('/Datasets/Insurance.csv')

#Basic EDA
df_insurance.head()
df_insurance.describe()
df_insurance.info()

#Separate the dependent and independent variables
X = df_insurance.iloc[:,0:1].values
y = df_insurance.iloc[:,1].values

print(type(X)) #-->numpy.ndarray
print(type(y)) #-->numpy.ndarray

print(X.shape) # (10,1)
print(y.shape) # (10,)

#LINEAR REGRESSION
#Fit on Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X,y) #LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)

lin_reg.score(X,y) #0.6690412331929894

y_pred = lin_reg.predict(X)
abs(y_pred) 
#array([11445.45454545,  3357.57575758,  4730.3030303 , 12818.18181818, 20906.06060606, 28993.93939394, 37081.81818182, 45169.6969697 , 53257.57575758, 61345.45454545])

#print('Age:', X[:,0]) #Age: [25 30 35 40 45 50 55 60 65 70]

#plotting the Linear Regression line
plt.scatter(X,y,color='red')
plt.plot(X,y_pred,color='blue')
plt.title('Insurance Premium Claim - Linear Regression')
plt.xlabel('Age')
plt.ylabel('Premium Claims')
plt.show()

#POLYNOMIAL REGRESSION
#The degree of the polynomial features. Default = 2.
#Heuristically change the degree value and see which is the best fit for the model
#If degree=3, X^1,X^2,X^3 will be computed and degree-1 columns are created
poly_features = PolynomialFeatures(degree=3)
poly_matrix = poly_features.fit_transform(X)

#Fit the polynomial features into linear regression
lin_reg_poly = LinearRegression()
lin_reg_poly.fit(poly_matrix,y) #LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)

y_pred_poly = lin_reg_poly.predict(poly_features.fit_transform(X))
lin_reg_poly.score(poly_matrix,y) #0.9812097727913366

#Plotting the Polynomial Regression line
plt.scatter(X,y,color='red')
plt.plot(X,y_pred_poly,color='green')
plt.title('Insurance Premium Claim - Polynomial Regression')
plt.xlabel('Age')
plt.ylabel('Premium Claims')
plt.show()

#Testing both the regressions with a test value and see what both regressors are predicting
#obviously for the predictions, the polynomial regression must be giving more accurate prediciton
new_age = 32
lin_pred = lin_reg.predict(new_age)
poly_pred = lin_reg_poly.predict(poly_features.fit_transform(new_age))

print('Linear Regression prediction for Age=32',abs(lin_pred)) #Linear Regression prediction for Age=32 [122.42424242]
print('Polynomial Regression prediction for Age=32', poly_pred) #Polynomial Regression prediction for Age=32 [8957.46386941]



