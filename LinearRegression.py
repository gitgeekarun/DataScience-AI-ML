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
