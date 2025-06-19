import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset=pd.read_csv('electricity.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
plt.figure(figsize=(10,5))
plt.plot(y_test,color='green')
plt.plot(y_pred,color='blue')
plt.show()
from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred))