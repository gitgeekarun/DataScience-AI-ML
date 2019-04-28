#Problem Statement Scenario:
#Since the first automobile, the Benz Patent Motor Car in 1886, Mercedes-Benz has stood for important automotive innovations. 
#These include the passenger safety cell with the crumple zone, the airbag, and intelligent assistance systems. 
#Mercedes-Benz applies for nearly 2000 patents per year, making the brand the European leader among premium carmakers. 
#Mercedes-Benz cars are leaders in the premium car industry. 
#With a huge selection of features and options, customers can choose the customized Mercedes-Benz of their dreams.
#To ensure the safety and reliability of every unique car configuration before they hit the road, Daimler’s engineers have developed a robust testing system. 
#As one of the world’s biggest manufacturers of premium cars, safety and efficiency are paramount on Daimler’s production lines. 
#However, optimizing the speed of their testing system for many possible feature combinations is complex and time-consuming without a powerful algorithmic approach.
#It is required to reduce the time that cars spend on the test bench. 
#Others will work with a dataset representing different permutations of features in a Mercedes-Benz car to predict the time it takes to pass testing. 
#Optimal algorithms will contribute to faster testing, resulting in lower carbon dioxide emissions without reducing Daimler’s standards. 

#import required libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,mean_squared_error,r2_score
import xgboost as xgb

xgb.__version__

#Import the train and test datasets
#Refer to the Datasets repository for actual files
df_train = pd.read_csv('mercedes_train.csv')
df_test = pd.read_csv('mercedes_test.csv')

#Do some basic EDA to check the dataset
print(df_train.info())
print(df_train.shape)
df_train.head()

#There is an ID column in the dataset which will be removed as it wont contribute to the prediction.
df_train = df_train.drop(columns=['ID'],axis=1)
df_test = df_test.drop(columns=['ID'],axis=1)

#Assigning the column 'y' to a new variable.
#The 'y' column is the predictor label.
#And drop the 'y' column from training set
df_labels = df_train['y']
df_train = df_train.drop(columns=['y'],axis=1)

print(df_train.shape) # Current shape (4209, 376)
print(df_test.shape) # Current shape (4209, 376)
print(df_labels.shape) # Current shape (4209, 1)

#Find the variance of all the features.
df_train_variance = pd.DataFrame(df_train.var(),columns=['variance'])
#Filter out the features which has variance=0
variance_filter = df_train_variance.loc[df_train_variance.variance == 0]

variance_filter.count()
#Find the indexes, so that these values can be removed.
variance_filter.index 
#Below are the 12 indexes which has 0 variance which will be removed from the data.
#Index(['X11', 'X93', 'X107', 'X233', 'X235', 'X268', 'X289', 'X290', 'X293','X297', 'X330', 'X347']

#Simply drop the indexes from both train and test
df_train_filter = pd.DataFrame(df_train.drop(columns=variance_filter.index))
df_test_filter = pd.DataFrame(df_test.drop(columns=variance_filter.index))

#Lets see the shapes of the old Vs new filtered datasets.
#There are 12 features with variance=0 has been removed.
print(df_train.shape) #(4209, 376)
print(df_test.shape) #(4209, 376)
print(df_train_filter.shape) #(4209, 364)
print(df_test_filter.shape) #(4209, 364)

df_train_filter.isnull().sum().count()

#Checking for categorical values in the dataset
df_train_filter.describe(include=['O'])
df_test_filter.describe(include=['O'])
#There are 8 of them which are object type/categorical X0,X1,X2,X3,X4,X5,X6 and X8

#Categorical Features to encode using LabelEncoder
categorical_col_to_encode = ['X0','X1','X2','X3','X4','X5','X6','X8']

#Create a LabelEncoder object
Lbl_encoder = LabelEncoder()

#Fitting and transforming the train and test data with LabelEncoder
for i in categorical_col_to_encode:
    Lbl_encoder.fit(df_train_filter[i].append(df_test_filter[i]).values)
    df_train_filter[i] = Lbl_encoder.transform(df_train_filter[i])
    df_test_filter[i] = Lbl_encoder.transform(df_test_filter[i])
    
#Now we do not have any categorical features in the datasets
#Check the details of the dtypes using the info() method
df_train_filter.info()
df_test_filter.info()

#See the shape of the datasets
print(df_train_final.shape) #(4209, 364)
print(df_test_final.shape) #(4209, 364)
print(df_labels.shape) #(4209,)

#splitting the train and labels (80/20 ratio)
xtrain,xtest,ytrain,ytest = train_test_split(df_train_final,df_labels,test_size=0.2,random_state=42)

#Now its time to reduce the dimensions of the datasets using PCA
#CONTINUE FROM HERE

