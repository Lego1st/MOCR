# -*- coding: UTF-8 -*-
from keras.models import load_model
from extract_chars import extract
from segmentation import *
from projection import project
import keras
import pickle
import numpy as np 
import sys
import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("-i", help="image path")
args = parser.parse_args()
im = cv2.imread(args.i)
reload(sys)
sys.setdefaultencoding("utf-8")

model = load_model('model1.h5')
dictionary = pickle.load(open('dictionary.pickle', "rb"))
projection = project(im)
chars = segment_single(im, projection)
word = []
for char in chars:
	cv2.imshow('predict',char)
	cv2.waitKey()
	print char.shape
	image = (char.astype(float) - 255.0/2)/255.0
	pred=np.ndarray.argmax(model.predict(image.reshape(1,30,30,1)))
	print dictionary[pred], model.predict(image.reshape(1,30,30,1))[0][pred]
	# if model.predict(image.reshape(1,30,30))[0][pred] > 0.3:
	word.append(dictionary[pred])
with open("Output.txt", "w") as text_file:
    text_file.write("Predict: %s" % ''.join(word))
