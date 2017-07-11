from keras.models import load_model, Model
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, Lambda, Input
from keras.models import Sequential
from keras.layers import LSTM, TimeDistributed, Bidirectional
from models import *
from keras import backend as K
from utils import *

import random
import keras
import pickle
import argparse
import numpy as np  
import os.path

ap = argparse.ArgumentParser()
ap.add_argument("-ba", required = True, type = int, help = "Batch Size")
ap.add_argument("-cl", required = True, type = int, help = "Number of classes")
ap.add_argument("-ep", required = True, type = int, help = "Number of epochs")
ap.add_argument("-lw", required = True, type = int, help = "Load weights")
ap.add_argument("-lr", required = True, type = float, help = "Learning rate")

args = ap.parse_args()

root = '/home/long/aeh16/train'
words = os.listdir(root)
words_dir = [os.path.join(root, f) for f in words]
samples = []
for w in words_dir:
    samples.extend([os.path.join(w, f) for f in os.listdir(w)])

labels = []
for i, w in enumerate(words):
    labels.extend([w]* len(os.listdir(words_dir[i])))

labels = [text_to_labels(l, forward_mapping) for l in labels]
dataset = zip(samples, labels)

print "%d training samples" % len(samples)
batch_size = args.ba
num_classes = args.cl
epochs = args.ep
img_rows, img_cols, img_channel = 32, 32, 3
input_shape = (img_rows, img_cols, img_channel)
learning_rate = args.lr

# model_cnn = CNN(input_shape, num_classes)
# pretrained_cnn = model_cnn.model
# pretrained_cnn.load_weights('cnn.h5')
# pretrained_cnn.summary()

M = CRNN(learning_rate)
M.model.summary()
if os.path.isfile('crnn_keras.h5') and args.lw == 1:
    M.model.load_weights('crnn_keras.h5')
    print "Loaded weights!"
for ep in range(epochs):
    loss_batch = []

    for batch in get_batches_crnn(dataset, batch_size = batch_size):
        loss, _  = M.train_step([batch[0], batch[1], batch[2], batch[3], batch[4], True])
        loss_batch.append(loss)

    print "Epoch %d: %f" % (ep, (float)(sum(loss_batch))/(len(loss_batch)))
    M.model.save_weights('crnn_keras.h5')

    r = np.random.randint(0, len(samples))
    for b in get_batches_crnn(zip([samples[r]], [labels[r]]), 1):
        p = M.predict_step([b[0], b[1], b[3], True])[0]
        print b[2]
    print "True: ", labels_to_text(labels[r])
    print "Predict: ", labels_to_text(p[0])

# for i in range(10):
#     r = np.random.randint(0, len(samples))
#     for b in get_batches_crnn(zip([samples[r]], [labels[r]]), 1):
#         p = M.predict_step([b[0], b[1], b[3], True])[0]
#         print b[2]
#     print "True: ", labels_to_text(labels[r])
#     print "Predict: ", labels_to_text(p[0])

# M = RNN(128, 63)
# if os.path.isfile('crnn_keras.h5') and args.lw == 1:
#     M.model.load_weights('crnn_keras.h5')
#     print "Loaded weights!"

# for ep in range(epochs):
#     loss_batch = []

#     for batch in get_batches(dataset, batch_size = batch_size):
#         loss, _  = M.train_step([to_features_vecs(batch[0], model_cnn), batch[1], batch[2], batch[3], True])
#         loss_batch.append(loss)

#     print "Epoch %d: %f" % (ep, (float)(sum(loss_batch))/(len(loss_batch)))
#     M.model.save_weights('crnn_keras.h5')

#     r = np.random.randint(0, len(samples))
#     for b in get_batches(zip([samples[r]], [labels[r]]), 1):
#         p = M.predict_step([to_features_vecs(b[0], model_cnn), b[2], True])[0]
#         print b[2]
#     print "True: ", labels_to_text(labels[r])
#     print "Predict: ", labels_to_text(p[0])



