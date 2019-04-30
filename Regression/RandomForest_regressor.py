#RANDOM FOREST REGRESSOR

#import required libraries

import numpy as np

3

import pandas as pd

4

import matplotlib.pyplot as plt

/anaconda3/lib/python3.6/importlib/_bootstrap.py:219: RuntimeWarning: numpy.dtype size changed, may indicate binary incompatibility. Expected 96, got 88
  return f(*args, **kwds)

1

#model

2

from sklearn.ensemble import RandomForestRegressor

1

#load the data as df using pandas

2

df_insurance = pd.read_csv('/Users/arun/Desktop/SimpliLearn-AI-Engineer-Certification/Machine Learning/Datasets/Insurance.csv')

3

​

Basic EDA

1

df_insurance.info()

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 10 entries, 0 to 9
Data columns (total 2 columns):
Age                                 10 non-null int64
Average Claims per Year (Rupees)    10 non-null int64
dtypes: int64(2)
memory usage: 240.0 bytes

1

df_insurance.describe()

	Age 	Average Claims per Year (Rupees)
count 	10.000000 	10.000000
mean 	47.500000 	24950.000000
std 	15.138252 	29937.388367
min 	25.000000 	4500.000000
25% 	36.250000 	6500.000000
50% 	47.500000 	13000.000000
75% 	58.750000 	27500.000000
max 	70.000000 	100000.000000
1

df_insurance.shape

(10, 2)

1

df_insurance.head()

	Age 	Average Claims per Year (Rupees)
0 	25 	4500
1 	30 	5000
2 	35 	6000
3 	40 	8000
4 	45 	11000
1

#Separate the dependent and independent values

2

X = df_insurance.iloc[:,0:1].values

3

y = df_insurance.iloc[:,1].values

1

print(type(X))

2

print(type(y))

<class 'numpy.ndarray'>
<class 'numpy.ndarray'>

1

print(X.shape)

2

print(y.shape)

(10, 1)
(10,)

1

#Fit RandomForest Regressor on X

2

RF_regressor = RandomForestRegressor(n_estimators=10,random_state=0)

1

RF_regressor.fit(X,y)

RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
           max_features='auto', max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=1, min_samples_split=2,
           min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
           oob_score=False, random_state=0, verbose=0, warm_start=False)

1

y_pred = RF_regressor.predict(X)

1

RF_regressor.score(X,y)

0.9704434230386582

1

plt.scatter(X,y,color='red')

2

plt.plot(X,y_pred,color='blue')

3

plt.title('Insurance Premium Claim - RF Regressor')

4

plt.xlabel('Age')

5

plt.ylabel('Premium Claims')

6

plt.show()

1

#Testing the regressor prediction on a unknown value

2

new_age = 100

3

pred = RF_regressor.predict(new_age)

4

print('RF Prediction:',pred)

RF Prediction: [85000.]
