# -*- coding: UTF-8 -*-
from PIL import Image, ImageDraw, ImageFont
from boundingDetect import fit_contours
import ttfquery.findsystem 
import ntpath
import numpy as np
import os
import glob
import sys
import cv2
reload(sys)
sys.setdefaultencoding("utf-8")

fontSize = 30
imgSize = (200,200)
position = (0,0)
 
#All images will be stored in 'Synthetic_dataset' directory under current directory
dataset_path = os.path.join (os.getcwd(), 'SVM_dataset')
if not os.path.exists(dataset_path):
	os.makedirs(dataset_path)
# exceptional_fonts = []
# for line in open('Stuff/fonts.txt', 'r'):
#    exceptional_fonts.append(line.rstrip('\n'))
lower_case_list = []
for line in open('Stuff/low.txt', 'r'):
   lower_case_list.append(line.rstrip('\n'))
upper_case_list = []
for line in open('Stuff/up.txt', 'r'):
   upper_case_list.append(line.rstrip('\n'))
digits = range(0,10)
digits_list=[]
for d in digits:
   digits_list.append(str(d))
 
all_char_list = lower_case_list + upper_case_list + digits_list
# print len(all_char_list)
# all_char_list = ['ổ']
# temp = [2,3,8,9,18,19,31,34,39,41,49]
# temp = temp +  map(lambda x: x+93, temp)
upper = [14,15,27,37,57,83,85,88,92]
lower = [76,80,81,82,84,86,87] + range(93, 186)
#paths = ttfquery.findsystem.findFonts()
all_fonts = glob.glob("/home/tailongnguyen/MOCR_test/Stuff/unicodeFonts/*.ttf")+glob.glob("/home/tailongnguyen/MOCR_test/Stuff/unicodeFonts/*.TTF")
# all_fonts = glob.glob("/home/tailongnguyen/MOCR_test/Stuff/fonts/*.ttf")+glob.glob("/home/tailongnguyen/MOCR_test/Stuff/fonts/*.TTF")
total = 0
for i, ch in enumerate(all_char_list): 
	save = dataset_path + "/"+ ch
	if os.path.exists(save):
		print "Skiping ", ch
		continue
	os.makedirs(save)
	print "Start processing ", ch, ":"
	number = 0
	for sys_font in all_fonts[:len(all_fonts)/5]:
	#print "Checking "+p
		font_file = ntpath.basename(sys_font)
		font_file = font_file.rsplit('.')
		font_file = font_file[0]
		#Check desired font
		print "Getting ", font_file
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
				draw.text(position, ch.decode('utf-8'), (0,0,0), font=font)
				l_u_d_flag = "u"
				if ch.islower():
					l_u_d_flag = "l"
				elif ch.isdigit():
					l_u_d_flag = "d"
				file_name = font_file + '_' + l_u_d_flag + '_' + ch + '_' + 'blur' + '_' + str(idx) + '.png'
				file_name = os.path.join(save,file_name)
				image.save(file_name)
				idx += 1
				img = cv2.imread(file_name)
				# cv2.imshow('test', img)
				# cv2.waitKey()
				img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
				b, l, t, r = fit_contours(img)
				# cv2.imshow('test', img)
				# cv2.waitKey()
				if t < b+1 and l < r+1:
					img1 = img[t:b+1, l:r+1]
					img1 = cv2.resize(img1, (30,30))
					cv2.imwrite(file_name, img1)
				if i in upper:
					if t-2 > 0:
						img2 = img[t-2:b+1, l:r+1]
						img2 = cv2.resize(img2, (30,30))
						cv2.imwrite(file_name[:-4]+'_2.png', img2)
						number +=1
					if t-4 > 0:
						img3 = img[t-4:b+1, l:r+1]
						img3 = cv2.resize(img3, (30,30))
						cv2.imwrite(file_name[:-4]+'_3.png', img3)
						number +=1
					if t-6 > 0:
						img4 = img[t-6:b+1, l:r+1]
						img4 = cv2.resize(img4, (30,30))
						cv2.imwrite(file_name[:-4]+'_4.png', img4)
						number +=1
				elif i in lower:
					if b+2 < img.shape[0]:
						img2 = img[t:b+2, l:r+1]
						img2 = cv2.resize(img2, (30,30))
						cv2.imwrite(file_name[:-4]+'_2.png', img2)
						number +=1
					if b+4 < img.shape[0]:
						img3 = img[t:b+4, l:r+1]
						img3 = cv2.resize(img3, (30,30))
						cv2.imwrite(file_name[:-4]+'_3.png', img3)
						number +=1
					if b+6 < img.shape[0]:
						img4 = img[t:b+6, l:r+1]
						img4 = cv2.resize(img4, (30,30))
						cv2.imwrite(file_name[:-4]+'_4.png', img4)
						number +=1
				else:
					if b+2 < img.shape[0] and t-2>0:
						img2 = img[t-2:b+2, l:r+1]
						img2 = cv2.resize(img2, (30,30))
						cv2.imwrite(file_name[:-4]+'_2.png', img2)
						number +=1
					if b+6 < img.shape[0] and t-6>0:
						img3 = img[t-6:b+6, l:r+1]
						img3 = cv2.resize(img3, (30,30))
						cv2.imwrite(file_name[:-4]+'_3.png', img3)
						number +=1
					if b+12 < img.shape[0] and t-12>0:
						img4 = img[t-12:b+12, l:r+1]
						img4 = cv2.resize(img4, (30,30))
						cv2.imwrite(file_name[:-4]+'_4.png', img4)
						number +=1
					if b+20 < img.shape[0] and t-20>0:
						img5 = img[t-20:b+20, l:r+1]
						img5 = cv2.resize(img5, (30,30))
						cv2.imwrite(file_name[:-4]+'_5.png', img5)
						number +=1
				# if i not in temp and all(map(lambda x: x not in font_file.lower(), exceptional_fonts)):
				# 	if l+3 < r-3:
				# 		img3 = img[t+1:b, l+3:r-3]
				# 		img3 = cv2.resize(img3, (30,30))
				# 		cv2.imwrite(file_name[:-4]+'_3.png', img3)
				# 		number +=1
				# 	if l+4 < r-4:
				# 		img4 = img[t+1:b, l+4:r-4]
				# 		img4 = cv2.resize(img4, (30,30))
				# 		cv2.imwrite(file_name[:-4]+'_4.png', img4)
				# 		number+=1


		# font = ImageFont.truetype(path, fontSize*2)
		# image = Image.new("RGB", map(lambda x: 2*x,imgSize), (255,255,255))
		# draw = ImageDraw.Draw(image)
		# pos_x = 20
		# pos_y = 20
		# position = (pos_x,pos_y)
		# idx = 0		
		# for y in [pos_y-1, pos_y]:
		# 	for x in [pos_x-1, pos_x]:
		# 		position = (x,y)
		# 		draw.text(position, ch.decode('utf-8'), (0,0,0), font=font)
		# 		l_u_d_flag = "u"
		# 		if ch.islower():
		# 			l_u_d_flag = "l"
		# 		elif ch.isdigit():
		# 			l_u_d_flag = "d"
		# 		file_name = font_file + '_' + l_u_d_flag + '_' + ch + '_' + str(idx) + '.png'
		# 		file_name = os.path.join(save,file_name)
		# 		image.save(file_name)
		# 		idx += 1
		# 		img = cv2.imread(file_name)
		# 		# cv2.imshow('test', img)
		# 		# cv2.waitKey()
		# 		img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		# 		b, l, t, r = fit_contours(img)
		# 		# cv2.imshow('test', img)
		# 		# cv2.waitKey()
		# 		if t < b+1 and l < r+1:
		# 			img1 = img[t:b+1, l:r+1]
		# 			img1 = cv2.resize(img1, (30,30))
		# 			cv2.imwrite(file_name, img1)
		# 		if t+1 < b-1 and l+1 < r-1:
		# 			img2 = img[t+1:b-1, l+1:r-1]
		# 			img2 = cv2.resize(img2, (30,30))
		# 			cv2.imwrite(file_name[:-4]+'_2.png', img2)
		# 		number +=2
		# 		if i not in temp and all(map(lambda x: x not in font_file.lower(), exceptional_fonts)):
		# 			if l+3 < r-3:
		# 				img3 = img[t+1:b-2, l+3:r-3]
		# 				img3 = cv2.resize(img3, (30,30))
		# 				cv2.imwrite(file_name[:-4]+'_3.png', img3)
		# 				number +=1
		# 			if l+4 < r-4:
		# 				img4 = img[t+1:b, l+4:r-4]
		# 				img4 = cv2.resize(img4, (30,30))
		# 				cv2.imwrite(file_name[:-4]+'_4.png', img4)
		# 				number +=1
	total += number
	print "Done processing ", ch, " with %d samples!" %number
print "Done with %d samples!" % total
				# gray = img
				# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
				# blur = cv2.GaussianBlur(gray,(3,3),0)
				# retVal, thresh = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
				# cv2.imwrite(file_name, thresh)
