from projection import project
from transformation import transform
import numpy as np 
import cv2
import argparse
import cv2
import matplotlib.pyplot as plt 
from boundingDetect import fit_contours

# parser = argparse.ArgumentParser()
# parser.add_argument("-i", help="image path")
# args = parser.parse_args()
# im = cv2.imread(args.i)
# gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
# retVal, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

alpha = 10
def isItalic(im):
	im_s = transform(im, alpha*np.pi/180)
	p, p_s = project(im), project(im_s)
	width = 1
	fig = plt.figure()
	a=fig.add_subplot(1,4,1)	
	plt.bar(range(len(p)), p, width, color="blue")
	a.set_title('Original')
	a=fig.add_subplot(1,4,2)
	plt.imshow(im)
	a=fig.add_subplot(1,4,3)
	plt.bar(range(len(p)), p_s, width, color="blue")
	a.set_title('Slanted')	
	a=fig.add_subplot(1,4,4)
	plt.imshow(im_s)
	plt.show()
	b,l,t,r = fit_contours(im)
	b_s, l_s, t_s, r_s = fit_contours(im_s)
	width = -l + r + 1
	width_s = -l_s + r_s +1
	print width, width_s
	if width < width_s:
		return False
	elif width > width_s:
		return True
	else:
		# print "Width unchanged"
		p, p_s = project(im), project(im_s)
		if max(p_s) > max(p):
			return False
		return True
# weight_roman = [0.8 , 0.5, 0.4]
# weight_italic = [0.15, 1, 1]

# def first_cri(im, im_s, p, p_s):
# 	print im.shape, im_s.shape
# 	width = 1
# 	fig = plt.figure()
# 	a=fig.add_subplot(1,4,1)	
# 	plt.bar(range(len(p)), p, width, color="blue")
# 	a.set_title('Original')
# 	a=fig.add_subplot(1,4,2)
# 	plt.imshow(im)
# 	a=fig.add_subplot(1,4,3)
# 	plt.bar(range(len(p)), p_s, width, color="blue")
# 	a.set_title('Slanted')	
# 	a=fig.add_subplot(1,4,4)
# 	plt.imshow(im_s)
# 	plt.show()
# 	if max(p_s) > max(p):
# 		return 1
# 	return 0
# def second_cri(im, im_s, p, p_s):
# 	space = len([i for i in p if i == 0])
# 	space_s = len([i for i in p_s if i == 0])
# 	if space_s > space:
# 		return 1
# 	return 0
# def third_cri(im, im_s, p, p_s):
# 	std = np.std(np.asarray(p))
# 	std_s = np.std(np.asarray(p_s))
# 	if std > std_s:
# 		return 1
# 	return 0

# def evaluate(alpha, im):
# 	im_s = transform(im, alpha)
# 	p, p_s = project(im), project(im_s)
# 	return first_cri(im, im_s, p, p_s), second_cri(im, im_s, p, p_s), third_cri(im, im_s, p, p_s)

# def validate(s, alpha, im):
# 	criterion = evaluate(alpha, im)
# 	print s, criterion
# 	score_roman = sum([criterion[i]*weight_roman[i] + (1-criterion[i])*(1-weight_roman[i]) for i in range(3)])
# 	score_italic =  sum([criterion[i]*weight_italic[i] + (1-criterion[i])*(1-weight_italic[i]) for i in range(3)])
# 	print "Score roman: ", score_roman
# 	print "Score italic: ", score_italic
# 	D = 'roman' if score_roman > score_italic else 'italic'
# 	if D == s:
# 		return 1
# 	return 0

# def classify(im):
# 	score_roman = sum([validate('roman', alpha*np.pi/180, im) for alpha in range(5,6)])
# 	score_italic = sum([validate('italic', alpha*np.pi/180, im) for alpha in range(5,6)])
# 	if score_italic > score_roman:
# 		return 'italic'
# 	return 'roman'


# print classify(thresh)

	


