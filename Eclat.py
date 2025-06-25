import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
ds=pd.read_csv('Netflix Movie.csv',header=None)
trans=[]
for i in range(7464):
    trans.append([str(ds.values[i][j]) for j in range(20) if str(ds.values[i][j]) != 'nan'])
print(trans)
from apyori import apriori
rules=apriori(transactions=trans,min_support=0.003,min_confidence=0.2,min_lift=3,min_length=2,max_length=2)

results=list(rules)

def inspect(results):
    product1=[tuple(result[2][0][0])[0] for result in results]
    product2=[tuple(result[2][0][1])[0] for result in results]
    support=[result[1] for result in results]
    return list(zip(product1,product2,support))

dataframe=pd.DataFrame(inspect(results),columns=['product1','product2','support'])
print(dataframe.nlargest(n=8,columns='support'))