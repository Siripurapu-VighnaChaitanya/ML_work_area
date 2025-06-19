import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
dataset=pd.read_csv('electricity.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor(random_state=0)
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
# np.set_printoptions(precision=2)
# print(np.concatenate((y_test.reshape(len(y_test),1),y_pred.reshape(len(y_pred),1)),axis=1))
plt.figure(figsize=(10,5))
plt.plot(y_test)
plt.plot(y_pred)
plt.show()
from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred))