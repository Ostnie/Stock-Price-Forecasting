import quandl, math
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import datetime

'''
We will forcast stock price by using a simple linear regression model
'''

#load data from Quandl for Google stock
df = quandl.get('WIKI/GOOGL')

#volatility  
df['HL_PCT'] = ((df['Adj. High'] - df['Adj. Low']) / df['Adj. Low']) * 100.0

#daily change in price
df['PCT_change'] = ((df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open']) * 100.0

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(value=-99999, inplace=True)
forecast_out = int(math.ceil(0.008 * len(df)))
print("Predict %s days out" %(forecast_out))

#adding label col
df['label'] = df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace=True)

y = np.array(df['label'])

print(len(y),len(X))



#Confrim shape by the folloing print statement:
#print(np.shape(X),np.shape(y))

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

#Creat clf instance
clf = LinearRegression()
#train your data using fit menthod
clf.fit(X_train,y_train)

'''
In order to pickle this clf, we run:
pickle_in = open('linearregression.pickle','rb')
clf = pickle.load(pickle_in)
'''

#check accuracy
#accuracy = clf.score(X_test, y_test)
#print(accuracy)

forecast_set = clf.predict(X_lately)
df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date (Years)')
plt.ylabel('Price (USD)')
plt.title("GOOGLE'S STOCK", {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 20,
        })
plt.show()