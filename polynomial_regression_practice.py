# impoeting libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing dataset

dataset=pd.read_csv('RBC in Humans Poly Reg.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

# Training the Linear Regression model on the whole dataset

from sklearn.linear_model import LinearRegression
linear_reg1=LinearRegression()
linear_reg1.fit(x,y)
y_pred_linear=linear_reg1.predict(x)

# Training the Polynomial Regression model on the whole dataset

from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=6)
x_poly=poly_reg.fit_transform(x)
linear_reg2=LinearRegression()
linear_reg2.fit(x_poly,y)
y_pred_poly=linear_reg2.predict(x_poly)

# Visualising the Linear Regression results

plt.scatter(x,y,color='red')
plt.plot(x,y_pred_linear,color='blue')
plt.title('RBC in Humans (Linear Regression)')
plt.xlabel('RBC')
plt.ylabel('Human')
plt.show()

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)

plt.scatter(x,y,color='green')
plt.plot(x,y_pred_poly,color='red')
plt.title('RBC in Humans (Polynomial Regression)')
plt.xlabel('RBC')
plt.ylabel('Human')
plt.show()

# Predicting a new result with Linear Regression

linear_reg1.predict([[9.5]])

# Predicting a new result with Polynomial Regression

linear_reg2.predict(poly_reg.fit_transform([[9.5]]))
