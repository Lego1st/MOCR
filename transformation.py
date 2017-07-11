import cv2
import numpy as np
import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument("-i", help="image path")
# args = parser.parse_args()
# im = cv2.imread(args.i)

def contrast(im):
	for i in range(im.shape[0]):
		for j in range(im.shape[1]):
			im[i][j] = 255 - im[i][j]
	return im

def transform(im, angle):
	rows,cols = im.shape[:2]	
	M = np.float32([[1,np.tan(angle),0],[0,1,0]])
	dst = cv2.warpAffine(contrast(im.copy()),M,(cols,rows))
	# blur = cv2.GaussianBlur(contrst(dst),(5,5),0)
	retVal, thresh = cv2.threshold(dst,127,255,cv2.THRESH_BINARY_INV)
	return thresh