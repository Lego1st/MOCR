from projection import project
from sklearn.cluster import KMeans
import numpy as np 
import argparse
import matplotlib.pyplot as plt 
import cv2
from transform import transform
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(parser.parse_args())

im = cv2.imread(args['image'])
gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
retVal, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

def segment(line):
	projection = project(line)
	SPACES = []
	space = []
	for i, p in enumerate(projection):
		if i != len(projection)-1:
			if p == 0:
				space.append(i)
			else:
				if len(space) > 0:
					SPACES.append((space[0], space[-1], len(space)))
					space = []
		else:
			if p == 0:
				space.append(i)
			if len(space) > 0:
				SPACES.append((space[0], space[-1], len(space)))

	x = np.ndarray((len(SPACES), 2), dtype = np.float)
	# print x
	for i in range(len(SPACES)):
		x[i] = [SPACES[i][-1],0]
	print x
	kmeans = KMeans(n_clusters=2, random_state=0).fit(x)
	cluster = kmeans.labels_
	space_chars = []
	space_words = []
	for i, c in enumerate(cluster):
		if c == 0:
			space_chars.append(SPACES[i][-1])
		else:
			space_words.append(SPACES[i][-1])
	if (float)(sum(space_chars))/len(space_chars) > (float)(sum(space_words))/len(space_words):
		cluster = [1 - c for c in cluster]

	boundary = []
	for i in range(len(SPACES)):
		if cluster[i] == 1:
			boundary.append((SPACES[i][1]+SPACES[i][0])/2)
	img = line.copy()
	for b in boundary:
		for i in range(img.shape[0]):
			img[i][b] = 0
	cv2.imshow('segmentation', img)
	cv2.waitKey()
	# width = 1
	# fig = plt.figure()
	# a=fig.add_subplot(1,2,1)	
	# plt.bar(range(len(projection)), projection, width, color="blue")
	# a.set_title('Vertical Projection')
	# a=fig.add_subplot(1,2,2)
	# plt.imshow(img)
	# a.set_title('Segmentation')	
	# plt.show()
	print boundary

segment(transform(thresh))