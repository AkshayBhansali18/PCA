import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset=pd.read_csv('C:\\Users\\aksha\\Documents\\ML documents\\PCA\\flowers.csv')
x=dataset.iloc[:,1:-1].values
y=dataset.iloc[:,-1].values
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x=sc.fit_transform(x)
from sklearn.decomposition import PCA
pca=PCA(n_components=2)
x_fin=pca.fit_transform(x)
pca1=[]
pca2=[]
print(x_fin)
for i in range(0,len(x_fin)):
    pca1.append(x_fin[i][0])
    pca2.append(x_fin[i][1])
print(pca.explained_variance_ratio_)
color_dict={"Iris-setosa":"red","Iris-versicolor":"green","Iris-virginica":"blue"}
i=0
for label in y:
    plt.scatter(pca1[i],pca2[i],c=color_dict[label])
    i=i+1
plt.show()

