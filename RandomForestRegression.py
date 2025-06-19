import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset=pd.read_csv('salary data1.csv')
X=dataset.iloc[:,1:-1].values
y=dataset.iloc[:,-1].values
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=10,random_state=25)
regressor.fit(X,y)
X_grid=np.arange(min(X),max(X),0.01)
X_grid=X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color='green',label='actual data')
plt.plot(X_grid,regressor.predict(X_grid),color='red',label='predict data')
plt.title('RandomForestRegression')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.legend()
plt.show()