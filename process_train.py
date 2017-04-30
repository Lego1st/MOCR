import cv2
import os
import argparse
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument("-root", help="train folder")
parser.add_argument("-save", help="save folder")
args = parser.parse_args()
def process_im(im, path, idx):
	gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
	kernel = np.ones((7,1), np.uint8)
	gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
	for i in range(gray.shape[0]):
		for j in range(gray.shape[1]):
			gray[i][j] = 255 - gray[i][j]
	im2, contours, hierarchy = cv2.findContours(gray,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

	def leftMost(x):
		x.sort(key=lambda x: x[0][0])
		return x[0][0][0]
	contours.sort(key=lambda x: leftMost(list(x)))
	count = 0	
	for i, cnt in enumerate(contours):
		if cv2.contourArea(cnt) > 20:
			x,y,w,h = cv2.boundingRect(cnt)
			if i > 0:
				x1,y1,w1,h1 = cv2.boundingRect(contours[i-1])
				if x >= x1 and x+w <= x1 + w1:
					continue
			count +=1
			if count > 1:
				return		
			# try:
			# 	count += 1
			# 	if count > 1:
			# 		raise Exception('Unexpected bound!')
			# except IOError as e:
			# 	print "Unexpected bound!", e, "It\'s okay, let\'s skip it!" 
			# image = cv2.rectangle(img, (x,0), (x+w,y+h), (0,255,0), 1)
			image = im[y:y+h, x:x+w, :]
			# pad = w/10
			# image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[255,255,255])
			image = cv2.resize(image,(30,30))
			gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
			blur = cv2.GaussianBlur(gray,(9,9),0)
			# thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,3,3)
			retVal, thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
			# cv2.imshow("image" + str(i) ,thresh)
			# cv2.waitKey(0)
			if not os.path.exists(path):
				os.makedirs(path)	
			cv2.imwrite(path + "/" + str(idx) + '.png', thresh)

def process(folder):
  print "Process: ", folder
  image_files = os.listdir(folder)
  num_images = 0
  temp = folder.split('/')[-1]
  for idx, image in enumerate(image_files):
    image_file = os.path.join(folder, image)
    im = cv2.imread(image_file)
    # try:
    process_im(im, save_folder + '/' + temp , idx)
    num_images +=1
    # except IOError as e:
    #   print "It\'s okay, let\'s skip it!" 
  print "Processed %d images!" % (num_images)
def get_fols(root):
	l = os.listdir(root)
	l = [root + "/" + x for x in l]
	return l

save_folder = args.save
train_folders = get_fols(args.root)

for folder in train_folders:
	process(folder)

