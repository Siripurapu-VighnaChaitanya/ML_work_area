import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data_set=pd.read_csv('salary data1.csv')
x=data_set.iloc[:,1:-1].values
y=data_set.iloc[:,-1].values
from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor(random_state=41)
regressor.fit(x,y)
#decision tree without clear view
plt.scatter(x,y,color='green',label='actual data')
plt.plot(x,regressor.predict(x),color='red',label='model predicted data')
plt.title('without grid')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.legend()
plt.show()
#decision tree with clear view
x_grid=np.arange(min(x),max(x),0.01)
x_grid=x_grid.reshape(len(x_grid),1)
plt.scatter(x,y,color='green',label='actual data')
plt.scatter(x_grid,regressor.predict(x_grid),color='blue')
plt.plot(x_grid,regressor.predict(x_grid),color='red',label='model predicted data')
plt.title('with grid')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.legend()
plt.show()