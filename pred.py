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
ap.add_argument("-i", required = True, type = str, help = "Image")

args = ap.parse_args()

link = args.i
patches = [window for window in sliding_window(ndimage.imread(link), 10)]
padded_patches = np.asarray(pad_sequences(patches, padding='post', value=255))
img_rows, img_cols, img_channel = 32, 32, 3
input_shape = (img_rows, img_cols, img_channel)
model_cnn = CNN(input_shape, 62)
pretrained_cnn = model_cnn.model
pretrained_cnn.load_weights('cnn.h5')
M = RNN(128, 63)
M.model.load_weights('crnn_keras.h5')
print "Loaded weights!"
p = M.predict_step([to_features_vecs([padded_patches], model_cnn), [len(patches)], True])[0]
print len(patches)
print "Predict: ", labels_to_text(p[0])