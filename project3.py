import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset=pd.read_csv('electricity.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
y_train=y_train.reshape(-1,1)
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
sc_y=StandardScaler()
x_train=sc_x.fit_transform(x_train)
y_train=sc_y.fit_transform(y_train)
y_test=y_test.reshape(-1,1)
x_test=sc_x.transform(x_test)
y_test=sc_y.transform(y_test)
from sklearn.svm import SVR
regressor=SVR(kernel='rbf')
regressor.fit(x_train,y_train.ravel())
y_pred_scale=regressor.predict(x_test)
y_pred=sc_y.inverse_transform(y_pred_scale.reshape(-1,1))
plt.figure(figsize=(10,5))
plt.plot(sc_y.inverse_transform(y_test))
plt.plot(y_pred)
plt.show()
from sklearn.metrics import r2_score
print(r2_score(sc_y.inverse_transform(y_test),y_pred))