# -*- coding: UTF-8 -*-
from keras.models import load_model
from extract_chars import extract
from dynamic_segmentation import *
from characters_segmentation import *
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
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
reload(sys)
sys.setdefaultencoding("utf-8")

model = load_model('model2.h5')
dictionary = pickle.load(open('dictionary.pickle', "rb"))
projection = project(im)
# chars = segment_single(im, projection)
chars = find_characters(im)
word = []
prob = []
for char in chars:
	# cv2.imshow('predict',char)
	# cv2.waitKey()
	image = (char.astype(float) - 255.0/2)/255.0
	pred=np.ndarray.argmax(model.predict(image.reshape(1,30,30,1)))
	prob.append(model.predict(image.reshape(1,30,30,1))[0][pred])
	word.append(dictionary[pred])

w = (''.join(word)).lower()
print 'Predict: %s (%f)' % (w, sum(prob)/len(prob))
with open("Output.txt", "w") as text_file:
    text_file.write("Predict: %s" % w)
