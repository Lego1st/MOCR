from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import keras
import pickle
import argparse
import numpy as np 
ap = argparse.ArgumentParser()
ap.add_argument("-ba", required = True, type = int, help = "Batch Size")
ap.add_argument("-cl", required = True, type = int, help = "Number of classes")
ap.add_argument("-ep", required = True, type = int, help = "Number of epochs")
args = ap.parse_args()
batch_size = args.ba
num_classes = args.cl
epochs = args.ep

img_rows, img_cols, img_channel = 30, 30, 1

# data = pickle.load(open("mocr_scale.pickle", "rb"))
# x_train, y_train, x_valid, y_valid, x_test, y_test = data['train_dataset'], data['train_labels'],data['valid_dataset'], data['valid_labels'], data['test_dataset'], data['test_labels']
# x_train = np.load('mocr_train.npy')
# x_train = x_train.reshape(-1,30,30,1)
# y_train = np.load('mocr_train_labels.npy')
# x_valid = np.load('mocr_valid.npy')
# x_valid = x_valid.reshape(-1,30,30,1)
# y_valid = np.load('mocr_valid_labels.npy')
# x_test = np.load('mocr_test.npy')
# x_test = x_test.reshape(-1,30,30,1)
# y_test = np.load('mocr_test_labels.npy')

x_train = np.load('train.npy')
x_train = x_train.reshape(-1,30,30,1)
y_train = np.load('train_labels.npy')

x_valid = np.load('valid.npy')
x_valid = x_valid.reshape(-1,30,30,1)
# x_valid = np.concatenate((x_v, x_valid), axis=0)
y_valid = np.load('valid_labels.npy')
# y_valid = np.concatenate((y_v, y_valid), axis=0)

x_train = np.concatenate((x_train, x_valid), axis=0)
y_train = np.concatenate((y_train, y_valid), axis=0)

x_test = np.load('test.npy')
x_test = x_test.reshape(-1,30,30,1)
# x_test = np.concatenate((x_t, x_test), axis=0)
y_test = np.load('test_labels.npy')
# y_valid = np.concatenate((y_t, y_test), axis=0)

input_shape = (img_rows, img_cols, img_channel)
print(x_train.shape[0], 'train samples')
# print(x_valid.shape[0], 'valid samples')
print(x_test.shape[0], 'test samples')


# # convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
# y_valid = keras.utils.to_categorical(y_valid, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# model = Sequential()

# model.add(Conv2D(16, kernel_size=3, data_format='channels_last',
#                  activation='relu',
#                  input_shape=input_shape))
# model.add(MaxPooling2D(pool_size=2))
# model.add(Conv2D(32,3 ,activation='relu', data_format='channels_last'))
# model.add(MaxPooling2D(pool_size=2))
# model.add(Conv2D(64,3 ,activation='relu', data_format='channels_last'))
# model.add(MaxPooling2D(pool_size=2))
# model.add(Dropout(0.25))
# # model.add(Dropout(0.25))
# # print model.output_shape

# model.add(Flatten())
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.25))
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes, activation='softmax'))

# model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=keras.optimizers.Adadelta(),
#               metrics=['accuracy'])
# model = load_model('model2.h5')
# model.fit(x_train, y_train,
#           batch_size=batch_size,
#           epochs=epochs,
#           verbose=1,
#           validation_data=(x_test, y_test))
# score = model.evaluate(x_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])

# model.save('model2.h5') 

