import numpy as np 
import cv2
import argparse
import words_extract as WE
import color_quantization as CQ 
import sentences_segmentation as SS 
import select_sample as SSamp 
import color_layers as CL
import os
import shutil
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
ap.add_argument("-c", "--clusters", required = True, type = int,
	help = "# of clusters")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
quant = CQ.quantize(args["clusters"], image)

# gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# blur = cv2.GaussianBlur(gray_img,(5,5),0)
# bw_img = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,3,3)

try:
	shutil.rmtree("text")
except:
	print("no text folder")

try:
	shutil.rmtree("nontext")
except:
	print("no nontext folder")

os.mkdir("text")
os.mkdir("nontext")

SSamp.select(quant, "text")
SSamp.select(quant, "nontext")

text_samples = os.listdir("text")
nontext_samples = os.listdir("nontext")
nontext_clrs = []

for nontext_sample in nontext_samples:
	nontext_image = cv2.imread("nontext/" + nontext_sample)
	for i in range(nontext_image.shape[0]):
		for j in range(nontext_image.shape[1]):
			if not np.any(nontext_clrs == nontext_image[i][j]):
				nontext_clrs.append(nontext_image[i][j])

text_clrs = []

for text_sample in text_samples:
	text_image = cv2.imread("text/"+text_sample)
	text_color = CL.textLayers(text_image, nontext_clrs)
	print(text_color)
	if text_color is not None:
		if not np.any(text_clrs == text_color):
			text_clrs.append(text_color)


# gray_img = cv2.cvtColor(quant, cv2.COLOR_BGR2GRAY)
print(len(text_clrs))
bw_img = quant[:,:,0].copy()
for i in range(quant.shape[0]):
	for j in range(quant.shape[1]):
		bw_img[i][j] = 0 if np.any(text_clrs == quant[i][j]) else 255

# ret,thresh = cv2.threshold(gray_img,127,255,0)

# cv2.imshow("black and white", bw_img)
# cv2.waitKey()

lines = SS.segmentize(bw_img, space_size = 30)
# lines = sorted(lines, key=lambda tup: tup[1])

i = 0

try:
	shutil.rmtree("result")
except:
	print("no result folder")

os.mkdir("result")

for line in lines:
	x, y, xh, yh = line
	img_line = bw_img[y: yh, x: xh]
	clr_img = bw_img[y: yh, x: xh]
	i += 1
	WE.text_strings(str(i) + "-", clr_img, img_line, max_distance_ratio = 0.05)
