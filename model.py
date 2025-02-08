from PIL import Image
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
import os
import keras
import tensorflow as tf
import pickle

with open('train_x.pkl','rb') as file_1:
    train_x = pickle.load(file_1)
with open('train_y.pkl','rb') as file_2:
    train_y = pickle.load(file_2)
with open('test_x.pkl','rb') as file_3:
    test_x = pickle.load(file_3)
with open('test_y.pkl','rb') as file_4:
    test_y = pickle.load(file_4)








#CNN Model Building
model = Sequential()

model.add(Conv2D(64, kernel_size=(3, 3),padding='same',activation='relu',input_shape=(32,32,3) , kernel_initializer='he_uniform'))
model.add(BatchNormalization())


model.add(Conv2D(128, kernel_size=(3, 3),padding='same',activation='relu',input_shape=(32,32,3) , kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


model.add(Conv2D(128, kernel_size=(3, 3),padding='same',activation='relu',input_shape=(32,32,3) , kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))



model.add(Conv2D(256, kernel_size=(3, 3),padding='same',activation='relu',input_shape=(32,32,3) , kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))



model.add(Conv2D(256, kernel_size=(3, 3),padding='same',activation='relu',input_shape=(32,32,3) , kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))



model.add(Conv2D(512, kernel_size=(3, 3),padding='same',activation='relu',input_shape=(32,32,3) , kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(512, kernel_size=(3, 3),padding='same',activation='relu',input_shape=(32,32,3) , kernel_initializer='he_uniform'))
model.add(BatchNormalization())

model.add(Conv2D(512, kernel_size=(3, 3),padding='same',activation='relu',input_shape=(32,32,3) , kernel_initializer='he_uniform'))
model.add(BatchNormalization())

model.add(Conv2D(512, kernel_size=(3, 3),padding='same',activation='relu',input_shape=(32,32,3) , kernel_initializer='he_uniform'))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(2, activation='softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer ='adam', metrics = ['accuracy'])
#data augmentation helps to reduce overfitting
train_generator = ImageDataGenerator(rotation_range=7, width_shift_range=0.05, shear_range=0.2,
                         height_shift_range=0.07, zoom_range=0.05)

test_genrator = ImageDataGenerator()

train_generator = train_generator.flow(train_x, train_y, batch_size=64)
test_generator = test_genrator.flow(test_x, test_y, batch_size=64)

model.fit_generator(train_generator, steps_per_epoch=100000//64, epochs=100,
                    validation_data=test_generator, validation_steps=20000//64)
"""
#old model
#Model Selection
model = Sequential()
model.add(Dense(1024 , input_dim = 1024 , activation = 'relu'))
model.add(Dense(1024 , activation = 'relu'))
model.add(Dense(512 , activation = 'relu'))
model.add(Dense(2 , activation = 'softmax'))
opt = keras.optimizers.Adam(lr=0.0003)
model.compile(loss = 'binary_crossentropy', optimizer = opt, metrics = ['accuracy'])
model.fit(features_train, labels_train, batch_size = 128, epochs = 100)
print(model.evaluate(features_train, labels_train))




model.evaluate(features_test, labels_test)
# Now 'data' holds your deserialized Python object"""


# Assume `model` is your trained TensorFlow model
model.save(r'C:\Users\rauna\OneDrive\Desktop\Projects\DeepFakeDetector\new_saved_model')  # Save in SavedModel format