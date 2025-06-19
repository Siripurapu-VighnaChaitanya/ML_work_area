import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
dataset = pd.read_csv('salary_data.csv')
x=dataset.iloc[ : , :-1].values
y=dataset.iloc[:,-1].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
plt.scatter(x_train,y_train,color="red")
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.show()
#-----------------------------------------------
# print("X_train:\n", x_train)
# print("y_train:\n", y_train)
# print("Predicted y:\n", regressor.predict(x_train))
plt.scatter(x_test,y_test,color="blue")
plt.plot(x_train,regressor.predict(x_train),color="green")
plt.show()
# print("X_test:\n", x_test)
# print("y_test:\n", y_test)
# print("y_pred (predicted salaries):\n", y_pred)
