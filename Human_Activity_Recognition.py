

#importing the library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset1=pd.read_csv('train.csv')
dataset2=pd.read_csv('test.csv')
X_train=dataset1.iloc[:,0:561].values
y_train=dataset1.iloc[:,[562]].values
X_test=dataset2.iloc[:,0:561].values
y_test=dataset2.iloc[:,[562]].values

#Encoding categoricaal data
from sklearn.preprocessing import LabelEncoder
labelencoder_y=LabelEncoder()
y_train=labelencoder_y.fit_transform(y_train[:,0])
y_test=labelencoder_y.fit_transform(y_test[:,0])



#feature scaling
from sklearn import preprocessing
sc_x=preprocessing.StandardScaler()
X_train=sc_x.fit_transform(X_train)
X_test=sc_x.transform(X_test)

#Apply PCA
from sklearn.decomposition import PCA
pca=PCA(n_components=200,whiten=True)
X_train=pca.fit_transform(X_train)
X_test=pca.transform(X_test)
explained_variance=pca.explained_variance_ratio_



#splitting into train_dev
from sklearn.model_selection import train_test_split
X_train,X_dev,y_train,y_dev=train_test_split(X_train,y_train,test_size=0.3,random_state=0)

#Applying SVM
from sklearn.svm import SVC
classifier = SVC(decision_function_shape= 'ovr', kernel='linear', random_state=0)
classifier.fit(X_train,y_train)




#
# Fitting classifier to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=30,criterion='entropy',random_state=0)
classifier.fit(X_train,y_train)


#confusion matrix
from sklearn.metrics import confusion_matrix
y_pred=classifier.predict(X_train)
cm_train=confusion_matrix(y_train,y_pred)
y_pred=classifier.predict(X_dev)
cm_test=confusion_matrix(y_dev,y_pred)
y_pred=classifier.predict(X_test)
cm_test=confusion_matrix(y_test,y_pred)


#Accuracy score
accuracy_train=classifier.score(X_train,y_train)
accuracy_dev=classifier.score(X_dev,y_dev)
accuracy_test=classifier.score(X_test,y_test)

#printing of the accuracy score
print('Training accuracy=',accuracy_train)
print('Development accuracy=',accuracy_dev)
print('Testing accuracy=',accuracy_test)

#Apply PCA (Reduction in two dimension)
from sklearn.decomposition import PCA
pca=PCA(n_components=2,whiten=True)
X_train=pca.fit_transform(X_train)
X_test=pca.transform(X_test)
explained_variance=pca.explained_variance_ratio_

#constructing dictionary for different colors corresponding to different activities
key={}
key['laying']='magenta'
key['sitting']='red'
key['standing']='yellow'
key['walking']='blue'
key['walking_downstairs']='green'
key['walking_upstairs']='purple'

#visualising the training set
from matplotlib.colors import ListedColormap
X_set,y_set=X_train,y_train

for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],c=ListedColormap(('red','green','purple','magenta','blue','yellow'))(i),label=j)
    plt.title('Human Activity Recognition (Train Set)')
    plt.xlabel('first feature(After pca dim_red)')
    plt.ylabel('second feature(After pca dim_red)')
plt.legend(key)
    plt.show()
    
#visualising the cross validation set set
from matplotlib.colors import ListedColormap
X_set,y_set=X_dev,y_dev

for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],c=ListedColormap(('red','green','purple','magenta','blue','yellow'))(i),label=j)
    plt.title('Human Activity Recognition (Cross Validation Set)')
    plt.xlabel('first feature(After pca dim_red)')
    plt.ylabel('second feature(After pca dim_red)')
plt.legend(key)
plt.show()
    
#visualising the test set
from matplotlib.colors import ListedColormap
X_set,y_set=X_test,y_test

for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],c=ListedColormap(('red','green','purple','magenta','blue','yellow'))(i),label=j)
    plt.title('Human Activity Recognition (Test Set)')
    plt.xlabel('first feature(After pca dim_red)')
    plt.ylabel('second feature(After pca dim_red)')
plt.legend(key)
plt.show()
    
