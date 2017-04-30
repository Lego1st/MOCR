import numpy as np 
import cv2
import matplotlib.pyplot as plt 
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-i", help="image path")
args = parser.parse_args()
im = cv2.imread(args.i)

def extract(im):
	chars = []
	img = im.copy()
	gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray,(3,3),0)
	thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,3,3)
	# kernel = np.ones((1,1), np.uint8)
	# gray = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
	gray = thresh
	for i in range(gray.shape[0]):
		for j in range(gray.shape[1]):
			gray[i][j] = 255 - gray[i][j]
	cv2.imshow('blah1', gray)
	cv2.waitKey(0)
	im2, contours, hierarchy = cv2.findContours(gray,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

	def leftMost(x):
		x.sort(key=lambda x: x[0][0])
		return x[0][0][0]
	new_contours = []	
	contours.sort(key=lambda x: leftMost(list(x)))
	for i, cnt in enumerate(contours):
		if cv2.contourArea(cnt) > 10:
			new_contours.append(cnt)		
	del_idx = []
	for i, cnt in enumerate(new_contours):
		x,y,w,h = cv2.boundingRect(cnt)
		for j, cnt1 in enumerate(new_contours[(i+1):]):
			x1,y1,w1,h1 = cv2.boundingRect(cnt1)
			if x <= x1 and x+w >= x1 + w1:
				del_idx.append(j+i+1)
	# for i, cnt in enumerate(new_contours):
	# 	if i not in del_idx:
	# 		x,y,w,h = cv2.boundingRect(cnt)
	# 		image = cv2.rectangle(img, (x,0), (x+w,y+h), (0,255,0), 1)
	# 		cv2.imshow('blah', image)
	# 		cv2.waitKey(0)

	for i, cnt in enumerate(new_contours):
		if i not in del_idx:
			x,y,w,h = cv2.boundingRect(cnt)
			image = cv2.rectangle(img, (x,0), (x+w,y+h), (0,255,0), 1)
			cv2.imshow('blah', image)
			cv2.waitKey(0)
			image = im[y:y+h, x:x+w, :]
			if h/w > 3:
				pad = h/10
			else:
				pad = 0
			image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[255,255,255])
			image = cv2.resize(image,(30,30))
			gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
			blur = cv2.GaussianBlur(gray,(3,3),0)
			# thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,9,3)
			retVal, thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
			chars.append(thresh)
	return chars