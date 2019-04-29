#import required libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score,mean_squared_error
from math import sqrt

#Allows plot to appear directly in notebook
%matplotlib inline

#read the advertising data to dataframe
adv_df = pd.read_csv('Advertising.csv',index_col=0)

#visualize the relationship between features and sales using scatter plot
fig,axes = plt.subplots(1,3,sharey=True)
adv_df.plot(kind='scatter',x='TV',y='sales',ax=axes[0],figsize=(16,8))
adv_df.plot(kind='scatter',x='radio',y='sales',ax=axes[1])
adv_df.plot(kind='scatter',x='newspaper',y='sales',ax=axes[2])

#Working with the feature variable 'TV' against the sales to find any relationship between them
#Create X('TV') and Y('Sales') variables
X = adv_df[['TV']]
Y = adv_df[['sales']]

#Now import, instantiate and fit the data
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X,Y)

#print the intercept and coefficents
print('Intercept:',lm.intercept_)
print('Coefficient:',lm.coef_)

# y = mx + c (0.04753664*spend_dollar + 7.03259355)
#m/Coefficient = 0.04753664
#Y Intercept = 7.03259355
#x(spend_dollar) = Unknown Input variable
spend_dollar = 50000
prediction = 0.04753664 * spend_dollar + 7.03259355
print('For every $50,000 spend on TV adv increases the sales of:',prediction,'widgets.')
