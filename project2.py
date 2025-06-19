import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset=pd.read_csv('electricity.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=6)
from sklearn.linear_model import LinearRegression
x_train_poly=poly_reg.fit_transform(x_train)
x_test_poly=poly_reg.transform(x_test)
regressor=LinearRegression()
regressor.fit(x_train_poly,y_train)
y_pred=regressor.predict(x_test_poly)
plt.figure(figsize=(10,5))
plt.plot(y_test[:50])
plt.plot(y_pred[:50])
plt.show()
from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred))