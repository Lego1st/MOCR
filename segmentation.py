import cv2
import numpy as np 
import argparse
import keras
from keras.models import load_model
from sklearn import svm
from sklearn.externals import joblib
from getCandidates_touched import score_touched
from getCandidates_single import score_single
from projection import project
from getFeatures import getFeatures
from boundingDetect import fit_contours
import pickle

# parser = argparse.ArgumentParser()
# parser.add_argument("-i", help="image path")
# args = parser.parse_args()
# im = cv2.imread(args.i)

def segment_touched(im, projection):

	def getTable(im, candidates):
		m = np.zeros((len(candidates), len(candidates)))
		for i in range(len(candidates)):
			for j in range(i+1, len(candidates)):
				subImage = im[:, candidates[i]: candidates[j]+1]
				subImage = cv2.resize(subImage, (30,30))
				subImage = cv2.cvtColor(subImage,cv2.COLOR_BGR2GRAY)
				subImage = (subImage.astype(float) - 255.0/2)/255.0
				pred=np.ndarray.argmax(model.predict(subImage.reshape(1,30,30,1)))
				# print model.predict(subImage.reshape(1,30,30,1))
				m[i][j] = model.predict(subImage.reshape(1,30,30,1))[0][pred]
				# print dictionary[pred], m[i][j]
				# cv2.imshow('eval',subImage)
				# cv2.waitKey()
		return m
	# print getTable(im)		
	def extract(im, table, candidates):
		chars = []
		prob = [table[0][i] for i in range(len(candidates))]
		trace = [0]*len(candidates)
		trace[0] = -1
		for i in range(len(candidates)):
			for j in range(i):
				if prob[j] + table[j][i] > prob[i]:
					trace[i] = j
					prob[i] = prob[j] + table[j][i]
		end = len(candidates)-1
		segments = []
		while trace[end] >= 0:
			segments.append((candidates[trace[end]], candidates[end]))
			end = trace[end]
			pass
		segments.reverse()
		for s in segments:
			if s[1]+1-s[0] > im.shape[0]*0.3:
				temp = im[:, s[0]:s[1]+1]
				chars.append(temp)
		print segments
		return chars

	sc, all_candidates, candidates = score_touched(projection)
	if 0 not in candidates:
		candidates = [0] + candidates
	if im.shape[1]-1 not in candidates:
		candidates = candidates + [im.shape[1]-1]
	# chars = extract(getTable(im))
	return extract(im, getTable(im, candidates), candidates)

def segment_single(im, projection):
	candidates = score_single(projection)
	if 0 not in candidates and projection[0] > 0:
		candidates = [0] + candidates
	if im.shape[1]-1 not in candidates and projection[-1] > 0:
		candidates = candidates + [im.shape[1]-1]
	chars = []
	for i in range(len(candidates)-1):
		temp = im[:,candidates[i]:candidates[i+1]+1]
		temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
		b,l,t,r = fit_contours(temp)
		temp = cv2.resize(temp[t:b+1, l:r+1], (30,30))
		chars.append(temp)
	return chars


# clf = joblib.load('svm.pkl')
# model = load_model('model2.h5')
# dictionary = pickle.load(open('dictionary.pickle', "rb"))
# projection = project(im)

# chars = process_single(im, projection)
# for i,c in enumerate(chars):
# 		cv2.imshow(str(i), c)
# 		cv2.waitKey()

# img = im.copy()
# for c in candidates:
# 	for i in range(im.shape[0]):
# 		img[i][c] = 0
# width = 1
# fig = plt.figure()
# a=fig.add_subplot(1,4,1)	
# plt.bar(range(len(projection)), projection, width, color="blue")
# a.set_title('Vertical Projection')
# a=fig.add_subplot(1,4,2)
# plt.bar(range(len(projection)), sc, width, color='red')
# a.set_title('Boundary Candidates')
# a=fig.add_subplot(1,4,3)
# plt.bar(range(len(projection)), all_candidates, width, color='red')
# a.set_title('Final Candidates')
# a=fig.add_subplot(1,4,4)
# plt.imshow(img)
# a.set_title('On original image')
# plt.show()