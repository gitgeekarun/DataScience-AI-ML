#import required libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score,mean_squared_error
from math import sqrt

#Allows plot to appear directly in notebook
%matplotlib inline

#read the advertising data to dataframe
adv_df = pd.read_csv('Advertising.csv',index_col=0)
