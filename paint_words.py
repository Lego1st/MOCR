import numpy as np 
import cv2
import os
from imgaug import augmenters as iaa
from boundingDetect import fit_contours

(winW, winH) = (5, 10)
folder = '/home/tailongnguyen/aeh16/text'
if not os.path.isdir(folder):
    os.mkdir(folder)

labels = list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')

contrast2 = iaa.ContrastNormalization((0.65, 0.85), per_channel=0.5)
add = iaa.Add(-40, per_channel=0.5)
noise = iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.08*255), per_channel=0.4)

def get_fols(di):
	l = os.listdir(di)
	l.sort()
	l = [di + "/" + x for x in l]
	return l
# stop = 0
contrast = iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5)
for i, fol in enumerate(get_fols(folder)):
	print "Processing " + fol
	image_files = os.listdir(fol) 
	# stop += 1
	for j, image in enumerate(image_files):
		image_file = os.path.join(fol, image)
		im = cv2.imread(image_file)
		gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
		blur = cv2.GaussianBlur(gray,(3,3),0)
		retVal, thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
		b, l, t, r = fit_contours(thresh)
		im_1 = im[t:b+1, l:r+1]
		scale = (float)(32 )/ im_1.shape[0]
		print scale
		print im_1.shape[1]
		print int(im_1.shape[1] * scale)
		im_1 = cv2.resize(im_1, (int(im_1.shape[1] * scale), 32) )
		im_2 = cv2.GaussianBlur(im_1,(7,7),0)

		a1 = add.augment_image(im_1)
		a2 = add.augment_image(im_2)

		b1 = contrast2.augment_image(im_1)
		b2 = contrast2.augment_image(im_2)

		c1 = noise.augment_image(im_1)
		c2 = noise.augment_image(im_2)

		d1 = noise.augment_image(a2)
		d2 = noise.augment_image(b2)
		
		cv2.imwrite(fol + "/" + str(i) + "_" + "0" + "_" + str(j) + ".png", im_1)
		cv2.imwrite(fol + "/" + str(i) + "_" + "1" + "_" + str(j) + ".png", im_2)
		cv2.imwrite(fol + "/" + str(i) + "_" + "2" + "_" + str(j) + ".png", a1)
		cv2.imwrite(fol + "/" + str(i) + "_" + "3" + "_" + str(j) + ".png", a2)
		cv2.imwrite(fol + "/" + str(i) + "_" + "4" + "_" + str(j) + ".png", b1)
		cv2.imwrite(fol + "/" + str(i) + "_" + "5" + "_" + str(j) + ".png", b2)
		cv2.imwrite(fol + "/" + str(i) + "_" + "6" + "_" + str(j) + ".png", c1)
		cv2.imwrite(fol + "/" + str(i) + "_" + "7" + "_" + str(j) + ".png", c2)
		cv2.imwrite(fol + "/" + str(i) + "_" + "8" + "_" + str(j) + ".png", d1)
		cv2.imwrite(fol + "/" + str(i) + "_" + "9" + "_" + str(j) + ".png", d2)
		
		# os.remove(image_file)
	# if stop >= 1:
	# 	break




