# 1 Import packages and classes
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

import torch
from torch import nn

import math
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms

dataset = pd.read_excel(r'C:\Users\Micha\Desktop\python\what to do\Dataset.xlsx', sheet_name='Fullo1')
dataset.columns = ['Chemical_name', 'SMILES', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14',
                   '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29']
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
# Clustering
le = LabelEncoder()
Y2 = le.fit_transform(Y1)
Y4 = le.fit_transform(Y_test1)
Y3 = pd.DataFrame(Y2)
# Make standardization
scaler = StandardScaler()
scaled_Y = scaler.fit_transform(Y3)
le.classes_
n_clusters = len(le.classes_)

Y = np.array(Y2)
Y_validation=np.array(Y4)


print(np.shape(Y))
print(np.shape(Y_validation))

device = ""
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(device)

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

transform1 = transforms.Compose(
    [transforms.ToTensor()]
)

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train = 'C:/Users/Micha/Desktop/python/jpgs/train'
train_dir = os.path.join(train)
validation = 'C:/Users/Micha/Desktop/python/jpgs/validation'
validation_dir = os.path.join(validation)

train_imgs = ['C:/Users/Micha/Desktop/python/jpgs/train/{}'.format(i) for i in os.listdir(train_dir)]
validation_imgs = ['C:/Users/Micha/Desktop/python/jpgs/validation/{}'.format(i) for i in os.listdir(validation_dir)]
nrows = 150
ncolumns = 150
channels = 3


def read_and_process_image(list_of_images):
    X = []
    for image in list_of_images:
        X.append(cv2.resize(cv2.imread(image, cv2.IMREAD_COLOR), (nrows, ncolumns), interpolation=cv2.INTER_CUBIC))
    return X


X = read_and_process_image(train_imgs)
X_validation = read_and_process_image(validation_imgs)

X = np.array(X)
X_validation = np.array(X_validation)

X1 = X

print(np.shape(X))
print(np.shape(X_validation))
X_tensor = X
X_t = X[0, :]
np.shape(X[0, :])
X_tensor = transform1(X_t)
a = []
k = 0
b = torch.zeros(349, 3, 150, 150)
for i in range(np.shape(X)[0]):
    a.append(transform(X[i, :]))
    b[i, :] = transform(X[i, :])
    k = k + 1
b.__sizeof__()
a.__sizeof__()
train_set = b

batch_size = 349
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True
)

real_samples = next(iter(train_loader))
mnist_labels = Y


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(67500, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.view(x.size(0), 67500)
        output = self.model(x)
        return output


discriminator = Discriminator().to(device=device)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(10000, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 67500),
            nn.Tanh(),
        )

    def forward(self, x):
        output = self.model(x)
        output = output.view(x.size(0), 3, 150, 150)
        return output


generator = Generator().to(device=device)

lr = 0.0001
num_epochs = 50
loss_function = nn.BCELoss()

optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)

for epoch in range(num_epochs):
    for n, (real_samples) in enumerate(train_loader):
        # Data for training the discriminator
        real_samples = real_samples.to(device=device)
        real_samples_labels = torch.ones((batch_size, 1)).to(
            device=device
        )
        latent_space_samples = torch.randn((batch_size, 10000)).to(
            device=device
        )
        generated_samples = generator(latent_space_samples)
        generated_samples_labels = torch.zeros((batch_size, 1)).to(
            device=device
        )
        all_samples = torch.cat((real_samples, generated_samples))
        all_samples_labels = torch.cat(
            (real_samples_labels, generated_samples_labels)
        )

        # Training the discriminator
        discriminator.zero_grad()
        output_discriminator = discriminator(all_samples)
        loss_discriminator = loss_function(
            output_discriminator, all_samples_labels
        )
        loss_discriminator.backward()
        optimizer_discriminator.step()

        # Data for training the generator
        latent_space_samples = torch.randn((batch_size, 10000)).to(
            device=device
        )

        # Training the generator
        generator.zero_grad()
        generated_samples = generator(latent_space_samples)
        output_discriminator_generated = discriminator(generated_samples)
        loss_generator = loss_function(
            output_discriminator_generated, real_samples_labels
        )
        loss_generator.backward()
        optimizer_generator.step()

        # Show loss
        if n == batch_size - 1:
            print(f"Epoch: {epoch} Loss D.: {loss_discriminator}")
            print(f"Epoch: {epoch} Loss G.: {loss_generator}")

latent_space_samples = torch.randn(batch_size, 10000).to(device=device)
generated_samples = generator(latent_space_samples)
generated_samples = generated_samples.cpu().detach()

for i in range(16):
    ax = plt.subplot(4, 4, i + 1)
    plt.imshow(torchvision.transforms.ToPILImage()(generated_samples[i]), cmap="gray_r")
    plt.xticks([])
    plt.yticks([])

for i in range(16):
    ax = plt.subplot(4, 4, i + 1)
    plt.imshow(generated_samples[i].reshape(150, 150, 3), cmap="gray_r")
    plt.xticks([])
    plt.yticks([])

X_right = torch.zeros(349, 150, 150, 3)
for i in range(349):
    X_right[i] = generated_samples[i].reshape(150, 150, 3)


X_tr = torch.zeros(279, 150, 150, 3)
for i in range(279):
    X_tr[i] = generated_samples[i].reshape(150, 150, 3)

X_traine = X_right.detach().numpy()[:, :, :, :]

X_train2 = X_traine[0:349, :, :, :]
X_val2 = X_traine[279:349, :, :, :]

X_train1 = X[0:349]/255
X_train = np.concatenate([X_train1, X_train2])

ntrain=len(X_train)

X_val1 = X[279:349]/255
X_val = np.concatenate([X_val1, X_val2])

nval = len(X_val)

Y_train1 = Y[0:349]
Y_train2 = Y_train1
Y_train = np.concatenate([Y_train1, Y_train2])
k=0
l=Y_train[0]
for i in range(349):
     if Y_train[i]!=l:
         l=Y_train[i]
         k=k+1
print(k)
classes = np.unique(Y_train)
nClasses = len(classes)

Y_val1 = Y[279:349]
Y_val2 = Y_val1
Y_val = np.concatenate([Y_val1, Y_val2])
k=0
l=Y_val[0]
for i in range(70):
     if Y_val[i]!=l:
         l=Y_val[i]
         k=k+1

print(k)

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


model = models.Sequential()
model.add(layers.Conv2D(32,(5,5),activation='relu', input_shape=(150,150,3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (5, 5), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(336, activation='softmax'))

model.compile(loss='categorical_crossentropy',
               optimizer='sgd',
               metrics=['accuracy'])

Y_train = np_utils.to_categorical(Y_train)
Y_val = np_utils.to_categorical(Y_val)
Y_validation2 = np_utils.to_categorical(Y_validation)
l=0
k=0
for i in range(698):
     if Y_train[i].any()==1:
         k=k+1
         l=1
k=0
l=0
for i in range(140):
     if Y_val[i].any()==1:
         k=k+1
         l=1


model.fit(X_train, Y_train,
           batch_size=100,
           epochs=5,
           verbose=1)

history=model.fit(X_train, Y_train,
           batch_size=100,
           epochs=5,
           verbose=1)

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
print(pred)
print(Y_validation2)
