import pandas as pd # for data manupulation or analysis
import numpy as np # for numeric calculation
import matplotlib.pyplot as plt # for data visualization
import seaborn as sns # for data visualization
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from tensorflow import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import Sequential
from keras.layers import Flatten, Dense, Dropout, BatchNormalization
from  keras.layers import MaxPool1D, Conv1D
import pickle
print(tf.__version__)

from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from  sklearn.preprocessing import StandardScaler

#Load breast cancer dataset
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print(cancer.DESCR)
print('Chech Point')
print(type(cancer))
cancer.keys()
# featurs of each cells in numeric format
x = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)
print(x.head(5))
y = cancer.target
print(y)
print(x.shape)
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=0, stratify=y)
print(x_train.shape)
print(x_test.shape)
#same scale
scaler = StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)
#converting into 3D array
x_train=x_train.reshape(455,30,1)
x_test=x_test.reshape(114,30,1)
print(x_train)
print(x_test)

epochs = 100

print('Building CNN Model')
model = Sequential()
model.add(Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=(30,1)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Conv1D(filters=128, kernel_size=2, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# model.add(Conv1D(filters=256, kernel_size=2, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))
print(model.summary())
print('Model Compilation')
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.00005), loss='binary_crossentropy', metrics=['accuracy'])

earlystopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1, mode='auto')
history = model.fit(x_train,y_train, epochs=epochs, validation_data=(x_test, y_test), verbose=1, callbacks=[earlystopper])
history_dict = history.history
print('Roger That')

model.save('model/Brest_Cancer.h5')
# checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
# callbacks_list = [checkpoint]

def plot_learningCurve(history, epoch):
    # summarize history for accuracy
    epoch_range = range(1, epoch+1)
    plt.plot(epoch_range, history_dict['accuracy'])
    plt.plot(epoch_range, history_dict['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(epoch_range, history_dict['loss'])
    plt.plot(epoch_range, history_dict['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

history.history
plot_learningCurve(history, epochs)
print('Ok Roger')
