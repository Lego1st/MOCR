# -*- coding: UTF-8 -*-
from keras.models import load_model
from dynamic_segmentation import *
from projection import project
from characters_segmentation import *
from preprocessing import preprocess 
import keras
import pickle
import numpy as np 
import sys
import argparse
import cv2


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", required = True, help = "Path to the image")
parser.add_argument("-c", "--clusters", required = True, type = int, help = "# of clusters")
args = vars(parser.parse_args())

im = cv2.imread(args['image'])
# im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
clusters = args['clusters'] 

reload(sys)
sys.setdefaultencoding("utf-8")

model = load_model('model2.h5')
dictionary = pickle.load(open('dictionary.pickle', "rb"))
# projection = project(im)
# chars = segment_single(im, projection)

with open("result/Output.txt", "w") as text_file:
    text_file.write("Predict: ")

words = preprocess(im, clusters)
for word in words:
	cv2.imshow('word',word)
	cv2.waitKey()
	chars = find_characters(word)
	c = []
	# prob = []
	for char in chars:
		image = (char.astype(float) - 255.0/2)/255.0
		pred=np.ndarray.argmax(model.predict(image.reshape(1,30,30,1)))
		# prob.append(model.predict(image.reshape(1,30,30,1))[0][pred])
		c.append(dictionary[pred])
		print 'Predict: %s (%f)' % (dictionary[pred], model.predict(image.reshape(1,30,30,1))[0][pred])

	w = (''.join(c)).lower()
	with open("result/Output.txt", "a") as text_file:
	    text_file.write(w + " ")
