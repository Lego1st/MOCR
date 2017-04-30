# -*- coding: UTF-8 -*-
from keras.models import load_model
from extract_chars import extract
from scipy import ndimage
import keras
import pickle
import numpy as np 
import matplotlib.pyplot as plt 
import sys
import argparse
import cv2
parser = argparse.ArgumentParser()
parser.add_argument("-i", help="image path")
args = parser.parse_args()

im = cv2.imread(args.i)
reload(sys)
sys.setdefaultencoding("utf-8")
model = load_model('new_model.h5')
dictionary = pickle.load(open('dictionary.pickle', "rb"))
# data = pickle.load(open("mocr.pickle", "rb"))
# x_test, y_test = data['test_dataset'], data['test_labels']
# pred = model.predict(x_test)
# print x_test.shape
chars = extract(im)
word = []
for char in chars:
	image = (char.astype(float) - 255.0/2)/255.0
	pred=np.ndarray.argmax(model.predict(image.reshape(1,30,30)))
	word.append(dictionary[pred])
with open("Output.txt", "w") as text_file:
    text_file.write("Predict: %s" % ''.join(word))