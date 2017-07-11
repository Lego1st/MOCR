# -*- coding: UTF-8 -*-
from PIL import Image, ImageDraw, ImageFont
from imgaug import augmenters as iaa
from boundingDetect import fit_contours
from scipy import ndimage
import ttfquery.findsystem 
import ntpath
import numpy as np
import os
import glob
import sys
import cv2

def paint_words(words_file = "/home/tailongnguyen/aeh16/eng.txt", folder= '/home/tailongnguyen/aeh16/train', font_fol = "/home/tailongnguyen/aeh16/unicodeFonts/" ):
	if not os.path.isdir(folder):
		os.mkdir(folder)

	(winW, winH) = (5, 10)
	fontSize = 30
	imgSize = (200,200)
	position = (0,0)

	contrast2 = iaa.ContrastNormalization((0.65, 0.85), per_channel=0.5)
	add = iaa.Add(-40, per_channel=0.5)
	noise = iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.08*255), per_channel=0.4)

	words_list = [ str(l).rstrip('\r\n') for l in open(words_file, 'r').readlines()]
	all_fonts = glob.glob(font_fol + "*")
	total = 0
	def togray(rgb):
		r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
		gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
		return gray
	for i, w in enumerate(words_list): 
		save = folder + "/"+ w
		if os.path.exists(save):
			print "Skiping ", w
			continue
		os.makedirs(save)
		print "Start processing %s: " % w
		for sys_font in all_fonts:
			font_file = ntpath.basename(sys_font)
			font_file = font_file.rsplit('.')
			font_file = font_file[0]
			#weck desired font
			path = sys_font
			font = ImageFont.truetype(path, fontSize)
			image = Image.new("RGB", imgSize, (255,255,255))
			draw = ImageDraw.Draw(image)
			pos_x = 10
			pos_y = 10
			position = (pos_x,pos_y)
			idx = 0
			for y in [pos_y-1, pos_y]:
				for x in [pos_x-1, pos_x]:
					position = (x,y)
					draw.text(position, w.decode('utf-8'), (0,0,0), font=font)
					file_name = font_file + '_' + w + '_' + str(0) + '.png'
					file_name = os.path.join(save,file_name)
					image.save(file_name)
					idx += 1
					im = ndimage.imread(file_name)
					gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
					# blur = cv2.GaussianBlur(gray,(3,3),0)
					# retVal, thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
					b, l, t, r = fit_contours(gray)
					im_1 = im[t:b+1, l:r+1]
					scale = (float)(32 )/ im_1.shape[0]
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
					
					cv2.imwrite(save + "/" + font_file + "_" + w + "_" + str(1) + ".png", togray(im_1))
					cv2.imwrite(save + "/" + font_file + "_" + w + "_" + str(2) + ".png", togray(im_2))
					cv2.imwrite(save + "/" + font_file + "_" + w + "_" + str(3) + ".png", togray(a1))
					cv2.imwrite(save + "/" + font_file + "_" + w + "_" + str(4) + ".png", togray(a2))
					cv2.imwrite(save + "/" + font_file + "_" + w + "_" + str(5) + ".png", togray(b1))
					cv2.imwrite(save + "/" + font_file + "_" + w + "_" + str(6) + ".png", togray(b2))
					cv2.imwrite(save + "/" + font_file + "_" + w + "_" + str(7) + ".png", togray(c1))
					cv2.imwrite(save + "/" + font_file + "_" + w + "_" + str(8) + ".png", togray(c2))
					cv2.imwrite(save + "/" + font_file + "_" + w + "_" + str(9) + ".png", togray(d1))
					cv2.imwrite(save + "/" + font_file + "_" + w + "_" + str(10) + ".png", togray(d2))
					
					os.remove(file_name)
			total += 11
			
	print "Done with %d samples!" % total

paint_words("/home/tailongnguyen/aeh16/eng.txt")