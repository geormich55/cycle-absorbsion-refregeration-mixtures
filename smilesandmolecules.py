#1 Import packages and classes
import gc
import os
import random
import zipfile
import cv2

from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy as np
import PIL
import PIL.Image
import tensorflow as tf
import pathlib
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.linear_model import LinearRegression
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from keras.utils import np_utils

#Import dataset
dataset = pd.read_excel(r'C:\Users\Micha\Desktop\python\what to do\Dataset.xlsx', sheet_name='Fullo1')
dataset.columns=['Chemical_name', 'SMILES', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29']
SMILES = dataset[['SMILES']]
MOLECULES = dataset[['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29']]
SMILES = SMILES.loc[3:448, :]
SMILES = SMILES.drop([5, 9, 13, 32, 35, 36, 39, 40, 46, 48, 69, 89, 94, 98, 99, 101, 109, 113, 118, 231, 249, 286, 306, 313, 314, 446, 447])
MOLECULES = MOLECULES.loc[3:448, :]
MOLECULES = MOLECULES.drop([5, 9, 13, 32, 35, 36, 39, 40, 46, 48, 69, 89, 94, 98, 99, 101, 109, 113, 118, 231, 249, 286, 306, 313, 314, 446, 447])
MOLECULES = pd.DataFrame(MOLECULES)
SMILES = pd.DataFrame(SMILES)
len(SMILES)
len(MOLECULES)
#Clustering
le = LabelEncoder()
scaled_smiles1 = le.fit_transform(SMILES)
scaled_smiles1 = pd.DataFrame(scaled_smiles1)
#Onehotencoding
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
X2  = ohe.fit_transform(scaled_smiles1).toarray()
# Make standardization
scaler = StandardScaler()
scaled_smiles = scaler.fit_transform(scaled_smiles1)
scaled_smiles=np.array(scaled_smiles)
MOLECULES=np.array(MOLECULES)
X=scaled_smiles.astype('float64')
Y=MOLECULES.astype(int)

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
Y2  = ohe.fit_transform(scaled_smiles1).toarray()

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(scaled_smiles1,Y,test_size = 0.1)
np.shape(y_train)
np.shape(y_test)
np.shape(X)
np.shape(X_train)
np.shape(X_test)

#X_train = np_utils.to_categorical(X_train)
#X_test = np_utils.to_categorical(X_test)

#An valeis X_train,Y2
#model = Sequential()
#model.add(Dense(5, input_dim=400, activation='relu'))
#model.add(Dense(3, activation='relu'))
#model.add(Dense(29, activation='softmax'))
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


model = Sequential()
model.add(Dense(5, input_dim=1, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(29, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=120, batch_size=10)
y_pred = model.predict(X_test)
print(y_pred.astype(int))
print(y_test)
#Converting predictions to label
pred = list()
for i in range(len(y_pred)):
  pred.append(np.argmax(y_pred[i]))
#Converting one hot encoded test label to label
test = list()
for i in range(len(y_test)):
  test.append(np.argmax(y_test[i]))
from sklearn.metrics import accuracy_score
a = accuracy_score(pred, test)
print('Accuracy is:', a * 100)
#plots
plt.plot(history.history['accuracy'])
#plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

history = model.fit(X_train, y_train,validation_data = (X_test,y_test), epochs=100, batch_size=10)
y_pred2 = model.predict(X_test)
print(y_pred2.astype(int))
print(y_test)
#Converting predictions to label
pred = list()
for i in range(len(y_pred)):
  pred.append(np.argmax(y_pred[i]))
#Converting one hot encoded test label to label
test = list()
for i in range(len(y_test)):
  test.append(np.argmax(y_test[i]))
from sklearn.metrics import accuracy_score
a = accuracy_score(pred, test)
print('Accuracy is:', a * 100)
#plots
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# create another model
#model = Sequential()
#model.add(Dense(5, input_dim=1, activation='relu'))
#model.add(Dense(3, activation='relu'))
#model.add(Dense(29, activation='softmax'))
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fix random seed for reproducibility
#seed = 6
#np.random.seed(seed)
# create model
#model = KerasClassifier(build_fn=model, epochs=150, batch_size=10, verbose=0)
# evaluate using 10-fold cross validation
#kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
#results = cross_val_score(model, X, Y, cv=kfold)
#print(results.mean())

