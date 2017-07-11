# -*- coding: UTF-8 -*-
from keras.models import load_model
from dynamic_segmentation import *
from projection import project
from characters_segmentation import *
from preprocessing import preprocess 
from lines_extraction import extract_lines
from words_extract import text_strings
from transform import transform
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
gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
retVal, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
clusters = args['clusters'] 
words = preprocess(im, clusters)

# reload(sys)
# sys.setdefaultencoding("utf-8")

# model = load_model('model2.h5')
# dictionary = pickle.load(open('dictionary.pickle', "rb"))
# projection = project(im)
# chars = segment_single(im, projection)

# with open("result/Output.txt", "w") as text_file:
#     text_file.write("Predict: ")

# stride = 5
# i = 1
# MAX_PROB = 0
# boudary = 0
# while i < thresh.shape[1]:
# 	img1 = thresh[:, :i].copy()
# 	img2 = thresh[:, i:].copy()
# 	char1 = find_characters(img1)
# 	char2 = find_characters(img2)
# 	prob  = []
# 	for j, char in enumerate(char1):
# 		M = 2
# 		# print "left" + str(i)
# 		# cv2.imshow('left' + str(i), char)
# 		# cv2.waitKey()
# 		image = (char.astype(float) - 255.0/2)/255.0
# 		pred=np.ndarray.argmax(model.predict(image.reshape(1,30,30,1)))
# 		p = model.predict(image.reshape(1,30,30,1))[0][pred]
# 		if (p <= M):
# 			M = p
# 		if j == len(char1)-1:
# 			prob.append(min(p, M))
# 		# print 'Predict: %s (%f)' % (dictionary[pred], model.predict(image.reshape(1,30,30,1))[0][pred])
# 	for j, char in enumerate(char2):
# 		M = 2
# 		# print "right" + str(i)
# 		# cv2.imshow('right' + str(i), char)
# 		# cv2.waitKey()
# 		image = (char.astype(float) - 255.0/2)/255.0
# 		pred=np.ndarray.argmax(model.predict(image.reshape(1,30,30,1)))
# 		p = model.predict(image.reshape(1,30,30,1))[0][pred]
# 		if (p <= M):
# 			M = p
# 		if j == len(char2)-1:
# 			prob.append(min(p, M))
# 		# print 'Predict: %s (%f)' % (dictionary[pred], model.predict(image.reshape(1,30,30,1))[0][pred])
# 	if sum(prob) >= MAX_PROB:
# 		MAX_PROB = sum(prob)
# 		boudary = i
# 	print "Sum: ", sum(prob), " at ", i
# 	i+=stride
# print MAX_PROB
# for i in range(thresh.shape[0]):
# 	thresh[i][boudary] = 0
# cv2.imshow('segmentation', thresh)
# cv2.waitKey()
# words = []
# lines = extract_lines(thresh)
# for i, l in enumerate(lines):
# 	words.extend(text_strings(str(i) + "-", l, l, max_distance_ratio = 0.2))
# words = [im]
# for word in words:
# 	# print "Showing at predict"
# 	cv2.imshow('word',word)
# 	cv2.waitKey()

# for word in words:
# 	# print "Showing at predict"
# 	cv2.imshow('word',word)
# 	cv2.waitKey()
# 	chars = find_characters(word)
# 	c = []
# 	# prob = []
# 	for char in chars:
# 		image = (char.astype(float) - 255.0/2)/255.0
# 		pred=np.ndarray.argmax(model.predict(image.reshape(1,30,30,1)))
# 		# prob.append(model.predict(image.reshape(1,30,30,1))[0][pred])
# 		c.append(dictionary[pred])
# 		print 'Predict: %s (%f)' % (dictionary[pred], model.predict(image.reshape(1,30,30,1))[0][pred])

# 	w = (''.join(c)).lower()
# 	with open("result/Output.txt", "a") as text_file:
# 	    text_file.write(w + " ")
