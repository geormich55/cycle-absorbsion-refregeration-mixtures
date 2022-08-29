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

#Import dataset
dataset = pd.read_excel(r'C:\Users\Micha\Desktop\python\what to do\Dataset.xlsx', sheet_name='Fullo1')
dataset.columns=['Chemical_name', 'SMILES', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29']
dat = dataset[['SMILES']]
cyclic = dat.loc[3:74, :]
cyclic1 = cyclic.drop([5, 9, 13, 32, 35, 36, 39, 40, 46, 48, 69])
straight = dat.loc[91:391, :]
straight1 = straight.drop([94, 98, 99, 101, 109, 113, 118, 231, 249, 286, 306, 313, 314])
cyclics = dat.loc[75:90, :]
cyclic2 = cyclics.drop([89])
str_cycl = dat.loc[392:448, :]
str_cycl2 = str_cycl.drop([446, 447])
frames1 = [cyclic1, straight1]
da = pd.concat(frames1)
frames2 = [cyclic2, str_cycl2]
data = pd.concat(frames2)
Y1 = pd.DataFrame(da)
Y_test1 = pd.DataFrame(data)
#Clustering
le = LabelEncoder()
Y2 = le.fit_transform(Y1)
Y3 = pd.DataFrame(Y2)
# Make standardization
scaler = StandardScaler()
scaled_Y = scaler.fit_transform(Y3)
le.classes_
n_clusters = len(le.classes_)
train = int(input("Enter the number of clusters that you want to disect the train values,the best is 3: "))
print(train)
# Run kmeans
kmeans = KMeans(
    init="random",
    n_clusters=train,
    n_init=10,
    max_iter=300,
    random_state=42
)
# Fit the results
kmeans.fit(scaled_Y)
# The lowest SSE value
kmeans.inertia_
# Final locations of the centroid
kmeans.cluster_centers_
# The number of iterations required to converge
kmeans.n_iter_
# kmeans labels
kmeans.labels_

Y=kmeans.labels_

Y_test2 = le.fit_transform(Y_test1)
Y_test3 = pd.DataFrame(Y_test2)
scaled_Y_test = scaler.fit_transform(Y_test3)

le.classes_
n_clusters = len(le.classes_)
test = int(input("Enter the number of clusters that you want to disect the train values,the best is 3: "))
print(test)
# Run kmeans
kmeans = KMeans(
    init="random",
    n_clusters=3,
    n_init=10,
    max_iter=300,
    random_state=42
)
# Fit the results
kmeans.fit(scaled_Y_test)
# The lowest SSE value
kmeans.inertia_
# Final locations of the centroid
kmeans.cluster_centers_
# The number of iterations required to converge
kmeans.n_iter_
# kmeans labels
kmeans.labels_

Y_validation=kmeans.labels_

train ='C:/Users/Micha/Desktop/python/jpgs/train'
train_dir=os.path.join(train)
validation='C:/Users/Micha/Desktop/python/jpgs/validation'
validation_dir=os.path.join(validation)

train_imgs=['C:/Users/Micha/Desktop/python/jpgs/train/{}'.format(i) for i in os.listdir(train_dir)]
validation_imgs=['C:/Users/Micha/Desktop/python/jpgs/validation/{}'.format(i) for i in os.listdir(validation_dir)]
nrows=150
ncolumns=150
channels=3
def read_and_process_image(list_of_images):
    X = []
    for image in list_of_images:
        X.append(cv2.resize(cv2.imread(image, cv2.IMREAD_COLOR), (nrows,ncolumns), interpolation=cv2.INTER_CUBIC))
    return X
X=read_and_process_image(train_imgs)
X_validation=read_and_process_image(validation_imgs)

X=np.array(X)
Y=np.array(Y)
X_validation=np.array(X_validation)
Y_validation=np.array(Y_validation)

print(np.shape(X))
print(np.shape(Y))
print(np.shape(X_validation))
print(np.shape(Y_validation))

from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.20, random_state=2)

ntrain = len(X_train)
nval = len(X_val)
batch_size = 10
classes = np.unique(Y_train)
nClasses = len(classes)

batch_size=32

#Make imports
from keras import layers
from keras import models
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img

#Build the model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(nClasses, activation='softmax'))

#Let's print the model summary
model.summary()

#Compile the model
from tensorflow.keras.optimizers import RMSprop
model.compile(optimizer=RMSprop(lr=0.001), loss='categorical_crossentropy', metrics = ['accuracy'])



Y_train = np_utils.to_categorical(Y_train)
Y_val = np_utils.to_categorical(Y_val)
Y_validation = np_utils.to_categorical(Y_validation)

history=model.fit(X_train, Y_train,
           validation_data = (X_val,Y_val),
           batch_size=10,
           epochs=50,
           verbose=1)

#history=model.fit(X_train, Y_train,
           #batch_size=10,
           #epochs=50,
           #verbose=1)

test_loss, test_acc = model.evaluate(X_val, Y_val)
print('Test accuracy:', test_acc)

#Save the model
model.save_weights('model_weights.h5')
model.save('model_keras.h5')

#plot the train and val curve
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

#Train and validation accuracy
plt.plot(epochs, acc, 'b', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and Validation accuracy')
plt.legend()

plt.figure()
#Train and validation loss
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()

plt.show()


pred = model.predict(X_validation/255)
print(pred.astype(int))
print(Y_validation)