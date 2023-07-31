# Importing the libraries

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from PIL import Image
import os
os.chdir('D:/Premswaroop/Traffic Sign Classification')
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

data = []
labels = []
# We have 43 Classes
classes = 43
cur_path = os.getcwd()
print(cur_path)

for i in range(classes):
    path = os.path.join(cur_path,'train',str(i))
    images = os.listdir(path)
    for a in images:
        try:
            image = Image.open(path + '\\'+ a)
            image = image.resize((30,30))
            image = np.array(image)
            data.append(image)
            labels.append(i)
        except Exception as e:
            print(e)

data = np.array(data)
labels = np.array(labels)

# os.mkdir('training')
np.save('./saved_models/data',data)
np.save('./saved_models/target',labels)

data=np.load('./saved_models/data.npy')
labels=np.load('./saved_models/target.npy')

print(data.shape, labels.shape)

# Dividing the dataset into Training and Testing
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=0)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# converting labels into one-hot encoding
y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)

# Building the model
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
# dropout layer to avoid overfitting
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
# We have 43 classes that's why we have defined 43 in the dense
# Softmax classifiers give you probabilities for each class label
model.add(Dense(43, activation='softmax'))

#Compilation of the model
# function that compares the target and predicted output values; measures how well the neural network models the training data.
# The purpose of an optimizer is to adjust model weights to maximize a loss function.
#
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 20
history = model.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_test, y_test))

def testing(testcsv):
    y_test = pd.read_csv(testcsv)
    label = y_test["ClassId"].values
    imgs = y_test["Path"].values
    data=[]
    for img in imgs:
        image = Image.open(img)
        image = image.resize((30,30))
        data.append(np.array(image))
    X_test=np.array(data)
    return X_test,label

X_test, label = testing('Test.csv')
Y_pred=model.predict(X_test)
print(Y_pred)
Y_pred=np.argmax(Y_pred,axis=1)
print(Y_pred)

# Evaluating the model's performance using appropriate metrics
from sklearn.metrics import accuracy_score
print(accuracy_score(label, Y_pred))                  #  model has achieved 94% accuracy

print('Classification Report: \n', classification_report(label, Y_pred))


model_version = 1
# saving the model in local storage
model.save(f"./saved_models/{model_version}")

