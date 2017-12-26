import pandas as pd
import quandl 

df = quandl.get('NSE/SAIL')

#To check the last values from the dataframe.
print(df.tail())

import pandas as pd
import quandl,math 
import numpy as np
from sklearn import preprocessing,cross_validation,svm
from sklearn.linear_model import LinearRegression

#replace 'NSE/SAIL' with any other stock name whose price you wan't to predict.
df = quandl.get('NSE/SAIL')
df = df[['Open','High','Low','Last','Close','Turnover (Lacs)','Total Trade Quantity']]
df['HL_PCT'] = (df['High'] - df['Low'])/df['Low']*100
df['PCT_Change'] = (df['Close'] - df['Open'])/df['Open']*100

df = df[['Close','HL_PCT','PCT_Change','Last','Turnover (Lacs)','Total Trade Quantity']]


forecast_col = 'Close'
df.fillna(-9999, inplace = True)

# .0002 corresponds to prediction of 1 day
# .0006 for 3 days
forecast_out = int(math.ceil(0.0006*len(df))) 


df['label'] = df[forecast_col].shift(-forecast_out)


X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X)
X = X[:-forecast_out]
X_lately = X[-forecast_out:]

df.dropna(inplace = True)
y = np.array(df['label'])
y = np.array(df['label'])

X_train , X_test , y_train , y_test = cross_validation.train_test_split(X , y , test_size =0.2 , random_state = 42)

clf = LinearRegression()
clf.fit(X_train , y_train)
accuracy = clf.score(X_test , y_test)

print(accuracy)

forecast_set = clf.predict(X_lately)

print(forecast_set , accuracy , forecast_out)
