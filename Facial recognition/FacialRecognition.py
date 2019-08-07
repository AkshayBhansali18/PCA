import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
#import dataset
dataset=pd.read_csv('C:\\Users\\aksha\\Documents\\ML documents\\Facial recognition\\FaceRecognition.csv')
#Split dataset into dependent and independent variables
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values
print(x.shape[1])
x=x.reshape(1,4096*42)
print(x.shape)
x[np.isnan(x)]=0
y[np.isnan(y)]=0
#Split data into training and test sets
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y)
print(x_train.shape)
pca=PCA(n_components=13,random_state=0,svd_solver='randomized')
xmod=pca.fit(x_train)
ratios=(pca.explained_variance_ratio_)
ratios=np.around(ratios,decimals=7)
x_train_pca=pca.transform(x_train)
from sklearn.svm import SVC
classifier=SVC(kernel='rbf',C=1000,gamma=0.01)
classifier.fit(x_train_pca,y_train)
x_test_pca=pca.transform(x_test)
y_pred=classifier.predict(x_test_pca)
from sklearn.metrics import confusion_matrix
cf=confusion_matrix(y_test,y_pred)

