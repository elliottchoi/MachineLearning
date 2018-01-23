import pandas as pd
import quandl
import math
import numpy as np
from sklearn import  preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
from audioop import cross
from scipy.linalg.tests.test_fblas import accuracy
from scipy.odr.models import polynomial
import matplotlib.pyplot as plt 
from matplotlib import style
import datetime
import pickle


style.use('ggplot')

df=quandl.get('WIKI/GOOGL')
df =df [['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume',]]

#High Low Change
df['HL_PCT']=((df['Adj. High']-df['Adj. Close']) / (df['Adj. Close']))*100

#Percent Daily Change
df['PCT_change']=(df['Adj. Close']-df['Adj. Open'])/df['Adj. Open']*100.0

#Think about the features that you choose 
#        price        x        x            x
df=df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume',]]

#where we are going to be putting in the regressed
forecast_col='Adj. Close'

#Cant work with NA in ML, replace NA with -9999 and it will be treated as an outlier 
df.fillna(-99999,inplace=True)

#integer value with ceil rounding , math.ceil returns a float 
#predict out 10% of the data frame
forecast_out=int(math.ceil(0.1*len(df)))
#we predict 34 days in advance
print forecast_out

#Shifting the Adj.Close, 10 days up, and placing it into label
df['label']=df[forecast_col].shift(-forecast_out)

#drop the entire dataframe except for the label column
x=np.array(df.drop(['label'],1))


#just make it label
y=np.array(df['label'])


#preprocessing by standarizing a dataset along any axis, center to the mean and component wise scale to unit variance
x=preprocessing.scale(x)

#stuff we predict against
x_lately=x[-forecast_out:]
x=x[:-forecast_out]

df.dropna(inplace=True)
#all the points we forecasted (1%)
y=np.array(df['label'])

#use 20% of training data to test 
x_train,x_test,y_train,y_test=cross_validation.train_test_split(x,y,test_size=0.2)

# #number of jobs refers to cpu dedication to processing
# clf=LinearRegression(n_jobs=10)
#  
# #train a classifier
# clf.fit(x_train, y_train)
# wb means to write binary, r means to read, rb means to read binary 
# #Pickle is used to save a classifier so that you do not need to re-train an algortihm
# with open ('linearregression.pickle','wb') as f:
#     #dump the classifier into f
#     pickle.dump(clf,f)

#reading the pickle (rb means to read)
pickler_in=open('linearregression.pickle','rb')
clf=pickle.load(pickler_in)

#score=test, train a classifier vs test, it will be 100% right
clf.score(x_test,y_test)

#accuracy of classifier
accuracy=clf.score(x_test, y_test,)
#print(accuracy)
forecast_set=clf.predict(x_lately)

#next 34 days of prices
print(forecast_set,accuracy,forecast_out)

#all values are na
df['Forecast']=np.nan
#name of last date
last_date=df.iloc[-1].name
#convert last date into unix 
last_unix=last_date.timestamp()
one_day_in_seconds=86400
#next day predicted values
next_unix=last_unix+one_day_in_seconds

for i in forecast_set:
    #Each forecast and day, set the values in dataframe, future features
    next_date=datetime.datetime.fromtimestamp(next_unix)
    next_unix+=one_day_in_seconds
    #All of the first columns are NA, i is the forecast from the regression model
    df.loc[next_date]=[np.nan for _ in range (len(df.columns)-1)]+[i]
#Checking
print df.head()
#Np.nan means all the stuff in range are "not a number", 
#+[i] is equal to the value in the forecast set
print df.tail()
#df.tail shows the second set of data, where, everything is NA but we fit in forecast as i after the last known date is finished 

df['Adj. Close'].plot()
df['Forecast'].plot()
#4th location is bottom right
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
# #Regression with SVM
# clf=svm.SVR(kernel='poly')
# #train a classifier
# clf.fit(x_train, y_train)
# 
# #score=test, train a classifier vs test, it will be 100% right
# clf.score(x_test,y_test)
# 
# #accuracy of classifier
# accuracy=clf.score(x_test, y_test,)
# print(accuracy)