from sklearn.cluster import MiniBatchKMeans
import cv2
import numpy as np 

def showImg(quant):
	cv2.imshow("xxx", quant)
	cv2.waitKey()
	
def quantize(n_clusters, image):
	"""
		Input: initial image
		Output: image with # (= # of clusters) colors 
	"""
	(h, w) = image.shape[:2]
	 
	image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
	 
	image = image.reshape((image.shape[0] * image.shape[1], 3))
	 
	clt = MiniBatchKMeans(n_clusters)
	labels = clt.fit_predict(image)
	quant = clt.cluster_centers_.astype("uint8")[labels]
	 
	quant = quant.reshape((h, w, 3))
	image = image.reshape((h, w, 3))
	 
	quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
	image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)


	return quant

def textLayers(quant):
	clrs = []
	for i in range(quant.shape[0]):
		for j in range(quant.shape[1]):
			if not np.any(clrs == quant[i][j]):
				clrs.append(quant[i][j])
	
	idx = 0
	new_img = quant.copy()
	final_quant = quant.copy()
	max_number_pix = 0
	for clr in clrs:
		for i in range(new_img.shape[0]):
			for j in range(new_img.shape[1]):
				if (quant[i][j] == clr).all():
					new_img[i][j] = np.array([255, 255, 255])
				else:
					new_img[i][j] = np.array([0, 0, 0])
		idx += 1
		print(idx)
		img = new_img[:,:,0]
		kernel	=	np.ones((1,new_img.shape[0]),	np.uint8) * 255
		img	=	cv2.dilate(img,	kernel,	iterations=1)
		img	=	cv2.erode(img,	kernel,	iterations=1)
		img	=	cv2.erode(img,	kernel,	iterations=1)

		cnt = 0
		for i in range(new_img.shape[0]):
			for j in range(new_img.shape[1]):
				if img[i][j] == 255:
					cnt += 1
		if cnt > max_number_pix:
			max_number_pix = cnt
			final_quant = new_img[:,:,0]
		# cv2.imshow("xxx", np.hstack([new_img[:,:,0]	, img]))
		# cv2.waitKey()
	return final_quant