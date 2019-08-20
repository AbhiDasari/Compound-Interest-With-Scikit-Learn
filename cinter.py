import pandas as pd
from sklearn import linear_model
data=pd.read_csv('cinter.csv')
print(data.head())
print(data.columns)
print(data.info)
regr = linear_model.LinearRegression()
import matplotlib.pyplot as plt
import numpy as np
x=data[['amount','rate']]
y=data['interest']
rgr=linear_model.LinearRegression()
rgr.fit(x,y)
print(rgr.intercept_)
print(rgr.coef_)
i=np.array([[6943.0,4.6]])

print(rgr.predict(i))
y_pred = rgr.predict(x)

plt.scatter(data['amount'],data['interest'])
plt.plot( data['amount'],y_pred, color='red')
plt.show()


