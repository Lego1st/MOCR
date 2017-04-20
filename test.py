from sklearn.cluster import MiniBatchKMeans
import numpy as np
import argparse
import cv2
 
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
ap.add_argument("-c", "--clusters", required = True, type = int,
	help = "# of clusters")
# ap.add_argument("-k", "--kernel", required = True, type = int,
# 	help = "size of kernel")
args = vars(ap.parse_args())


def color_quantization(args, image):
	"""
		Input: initial image
		Output: image with # (= # of clusters) colors 
	"""
	(h, w) = image.shape[:2]
	 
	image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
	 
	image = image.reshape((image.shape[0] * image.shape[1], 3))
	 
	clt = MiniBatchKMeans(n_clusters = args["clusters"])
	labels = clt.fit_predict(image)
	quant = clt.cluster_centers_.astype("uint8")[labels]
	 
	quant = quant.reshape((h, w, 3))
	image = image.reshape((h, w, 3))
	 
	quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
	image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
	return quant	
 
iterationCnt = 0
expandable_CC = 0

class CC:
	expandable = False
	


def testConditions(img, max_size_ratio, max_curvature_ratio):
	# 1st scan
	
	# 2nd scan

def countExpandableCC(img, max_distance_ratio):
	


def text_strings(img, max_size_ratio, max_curvature_ratio, max_distance_ratio):
	"""
		Input: binary image
	 	Output: images of strings
	"""
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			expandable_CC.append((i, j))

	while True:
		testConditions(img, max_size_ratio, max_curvature_ratio)
		countExpandableCC(img, max_distance_ratio)
		iterationCnt += 1
		if len(expandable_CC) == 0:
			break


	


image = cv2.imread(args["image"])
quant = color_quantization(args, image)

gray_img = cv2.cvtColor(quant, cv2.COLOR_BGR2GRAY)
(thresh, bw_img) = cv2.threshold(gray_img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

kernel = np.ones((1, 2), np.uint8)
new_img = cv2.erode(bw_img, kernel, iterations = 1)

cv2.imshow("image", np.hstack([bw_img, new_img]))
cv2.waitKey(0)