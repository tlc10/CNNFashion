# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import keras
import sklearn
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras import optimizers
from keras.callbacks import Callback
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import tensorflow as tf
from azureml.core import Run
from utils import load_data, one_hot_encode
import onnxmltools


print("Keras version:", keras.__version__)
print("Tensorflow version:", tf.__version__)

parser = argparse.ArgumentParser()
parser.add_argument('--data-folder', type=str, dest='data_folder', help='data folder mounting point')
parser.add_argument('--batch-size', type=int, dest='batch_size', default=50, help='mini batch size for training')
parser.add_argument('--learning-rate', type=float, dest='learning_rate', default=0.001, help='learning rate')

args = parser.parse_args()

data_folder = args.data_folder

print('training dataset is stored here:', data_folder)

(X_train,y_train),(X_test,y_test) = fashion_mnist.load_data()

X_train = X_train.reshape(60000,28,28,1)
X_test = X_test.reshape(10000,28,28,1)

training_set_size = X_train.shape[0]

n_epochs = 5
batch_size = args.batch_size
learning_rate = args.learning_rate

labelencoder = LabelEncoder()
y_train = labelencoder.fit_transform(y_train)
y_test = labelencoder.fit_transform(y_test)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


print(X_train.shape, y_train.shape, X_test.shape, y_test.shape, sep='\n')

# Build a simple MLP model
model = Sequential()
model.add(Conv2D(128, kernel_size=3, activation='relu', input_shape=(28,28,1)))
model.add(MaxPooling2D((2, 2),padding='same'))
model.add(Conv2D(64, kernel_size=3, activation='relu'))
model.add(MaxPooling2D((2, 2),padding='same'))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(MaxPooling2D((2, 2),padding='same'))
model.add(Flatten())
model.add(Dense(32,activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy','mse','mae'])

# start an Azure ML run
run = Run.get_context()


class LogRunMetrics(Callback):
    # callback at the end of every epoch
    def on_epoch_end(self, epoch, log):
        # log a value repeated which creates a list
        run.log('Loss', log['loss'])
        run.log('Accuracy', log['acc'])


history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=n_epochs,
                    verbose=2,
                    validation_data=(X_test, y_test),
                    callbacks=[LogRunMetrics()])

score = model.evaluate(X_test, y_test, verbose=0)

# log a single value
run.log("Final test loss", score[0])
print('Test loss:', score[0])

run.log('Final test accuracy', score[1])
print('Test accuracy:', score[1])

run.log('Final test mse', score[2])
print('Test mse:', score[2])

run.log('Final test mae', score[3])
print('Test mae:', score[3])


plt.figure(figsize=(6, 3))
plt.title('MNIST with Keras MLP ({} epochs)'.format(n_epochs), fontsize=14)
plt.plot(history.history['acc'], 'b-', label='Accuracy', lw=4, alpha=0.5)
plt.plot(history.history['loss'], 'r--', label='Loss', lw=4, alpha=0.5)
plt.plot(history.history['mean_squared_error'], 'y-', label='MSE', lw=4, alpha=0.5)
plt.plot(history.history['mean_absolute_error'], 'g--', label='MAE', lw=4, alpha=0.5)
plt.legend(fontsize=12)
plt.grid(True)

# log an image
run.log_image('Accuracy vs Loss vs MSE & MAE', plot=plt)

# create a ./outputs/model folder in the compute target
# files saved in the "./outputs" folder are automatically uploaded into run history
os.makedirs('./outputs/model', exist_ok=True)

# serialize NN architecture to JSON
model_json = model.to_json()
# save model JSON
with open('./outputs/model/model.json', 'w') as f:
    f.write(model_json)
# save model weights
model.save_weights('./outputs/model/model.h5')
print("model saved in ./outputs/model folder")


onnx_model = onnxmltools.convert_keras(model) 
onnxmltools.utils.save_model(onnx_model, './outputs/model/fashion.onnx')

