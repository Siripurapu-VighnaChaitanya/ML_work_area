import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
ds=pd.read_csv('amazon.csv')
x=ds.iloc[:,[2,4]].values
import scipy.cluster.hierarchy as sch
dendrogram=sch.dendrogram(sch.linkage(x,method='ward'))
plt.title('Dendrogram')
plt.xlabel('age')
plt.ylabel('rating')
plt.show()
from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=7,metric='euclidean',linkage='ward')
a_hc=hc.fit_predict(x)
plt.scatter(x[a_hc==0,0],x[a_hc==0,1],label='cluster 0')
plt.scatter(x[a_hc==1,0],x[a_hc==1,1],label='cluster 1')
plt.scatter(x[a_hc==2,0],x[a_hc==2,1],label='cluster 2')
plt.scatter(x[a_hc==3,0],x[a_hc==3,1],label='cluster 3')
plt.scatter(x[a_hc==4,0],x[a_hc==4,1],label='cluster 4')
plt.scatter(x[a_hc==5,0],x[a_hc==5,1],label='cluster 5')
plt.scatter(x[a_hc==6,0],x[a_hc==6,1],label='cluster 6')
plt.show()