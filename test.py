from sklearn.cluster import MiniBatchKMeans
import numpy as np
import argparse
import cv2
import math

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
 
BG_COLOR = 255
FG_COLOR = 0
EXPAND_CAND = 1
iterationCnt = 0
expandable_CC = 0

class CC:
	def __init__(self, expandable = False):
		self.expandable = expandable
		self.components = []
		self.connectedCC = []
		self.leftmost = 0
		self.rightmost = 0
		self.upmost = 0 
		self.downmost = 0
		self.beforeExpandSize = 0

	def bound_size(self):
		return max(self.rightmost - self.leftmost + 1, self.downmost - self.upmost + 1)

	def centroid(self):
		return int((self.leftmost + self.rightmost)/2), int((self.upmost + self.downmost)/2)

	def add(self, items):
		for item in items:
			if item in self.components:
				continue
			if not self.components:
				self.rightmost = self.leftmost = item[0]
				self.upmost = self.downmost = item[1]
			self.rightmost = max(self.rightmost, item[0])
			self.leftmost = min(self.leftmost, item[0])
			self.upmost = min(self.upmost, item[1])
			self.downmost = max(self.downmost, item[1])
			self.components.append(item)

	def resize(self):
		self.beforeExpandSize = self.bound_size()

CCs = []
expandable_CCs = []
finalWords = []

moves = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]

def stringCurvatureTest(img, i, j, max_curvature_ratio):
	return True

def passTest(img, isBG, idCC, x, y, max_size_ratio, max_curvature_ratio):
	passStringCurvatureTest = stringCurvatureTest(img, x, y, max_curvature_ratio)
	passConnectivityTest = False
	passSizeTest = False
	passExpandabilityTest = False
	chars = []



	for mv in moves:
		u, v = x + mv[0], y + mv[1]
		if u >= 0 and u < img.shape[0] and v >= 0 and v < img.shape[1]:
			if not isBG[u][v]:
				if idCC[u][v] not in chars:
					chars.append(idCC[u][v])
					if CCs[chars[-1]].expandable:
						passExpandabilityTest = True
	passConnectivityTest = len(chars) >= 1 and len(chars) <= 2
	if(len(chars) > 0):
		CCs[chars[-1]].add([(x, y)])
	if len(chars) == 2:
		sizeRation = max(CCs[chars[0]].beforeExpandSize, CCs[chars[1]].beforeExpandSize) \
					/ min(CCs[chars[0]].beforeExpandSize, CCs[chars[1]].beforeExpandSize)
		# CCs[chars[0]].add(CCs[chars[1]].components)
		# CCs[chars[1]].add(CCs[chars[0]].components)
		# CCs[chars[0]].connectedCC.append(chars[1])
		# CCs[chars[1]].connectedCC.append(chars[0])
		passSizeTest = sizeRation <= 2
	elif len(chars) == 1:
		passSizeTest = True

	res = passConnectivityTest and passSizeTest and passExpandabilityTest
	return (res, chars, BG_COLOR + 1 + chars[-1] if res else BG_COLOR) 



def testConditions(img, max_size_ratio, max_curvature_ratio):
	expansionCands = []
	# # 1st scan
	isBG = np.equal(img, [[BG_COLOR]*img.shape[1]] * img.shape[0])
	idCC = img - np.array([[BG_COLOR]*img.shape[1]] * img.shape[0]) - 1
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			if isBG[i][j]:
				isPass, connectedCCs, color = passTest(img, isBG, idCC, i, j, max_size_ratio, max_curvature_ratio)
				if isPass:
					expansionCands.append((i, j, color, connectedCCs))

	disqualifed = []
	# 2nd scan
	for i, j, c, ccs in expansionCands:
		img[i][j] = c
	isBG = np.equal(img, [[BG_COLOR]*img.shape[1]] * img.shape[0])
	idCC = img - np.array([[BG_COLOR]*img.shape[1]] * img.shape[0]) - 1

	for i, j, c, ccs in expansionCands:
		if not passTest(img, isBG, idCC, i, j, max_size_ratio, max_curvature_ratio):
			disqualifed.append((i, j))
		else:
			if len(ccs) == 2:
				CCs[ccs[0]].connectedCC.append(ccs[1])
				CCs[ccs[1]].connectedCC.append(ccs[0])

	for i, j in disqualifed:
		img[i][j] = BG_COLOR

def countExpandableCC(img, max_distance_ratio):
	global expandable_CC
	global iterationCnt
	expandable_CC = 0
	i = 0
	for cc in CCs:
		if len(cc.connectedCC) == 2 or iterationCnt > math.floor(max_distance_ratio  * cc.beforeExpandSize):
			cc.expandable = False
		else:
			expandable_CC += 1
		i += 1


def DFS(img, isBG, x, y, curCC, CC_id):
	# CC_id start from 1

	stack = [(x, y)]	
	
	while stack:
		x, y = stack.pop()

		img[x][y] = BG_COLOR+CC_id
		isBG[x][y] = True
		curCC.add([(x, y)])

		for mv in moves:
			u, v = x + mv[0], y + mv[1]
			if u >= 0 and u < img.shape[0] and v >= 0 and v < img.shape[1]:
				# print(type(img[u][v]))
				if not isBG[u][v]:
					stack.append((u, v))

def text_strings(img, max_size_ratio = 2.0, max_curvature_ratio = 0.3, max_distance_ratio = 0.05):
	"""
		Input: binary image
	 	Output: images of strings
	"""
	# cv2.imshow('black and white', img)
	# cv2.waitKey()
	# return
	global BG_COLOR
	global FG_COLOR

	test_img = img.copy()
	test_img = test_img.astype('int64')
	visited = []

	for i in range(test_img.shape[0]):
		for j in range(test_img.shape[1]):
			if test_img[i][j] != FG_COLOR:
				test_img[i][j] = BG_COLOR

	isBG = np.equal(test_img, [[BG_COLOR]*img.shape[1]] * img.shape[0])

	for i in range(test_img.shape[0]):
		for j in range(test_img.shape[1]):
			if not isBG[i][j]:
				new_CC = CC()
				DFS(test_img, isBG, i, j, new_CC, len(CCs)+1)
				# finalWords.append(len(CCs))
				new_CC.expandable = True
				new_CC.resize()
				CCs.append(new_CC)
	
	# for cc in CCs:
	# 	cv2.imshow("Char", img[cc.leftmost:cc.rightmost+1, cc.upmost:cc.downmost+1])
	# 	cv2.waitKey()
	global iterationCnt
	global expandable_CC

	iterationCnt = 0
	expandable_CC = 0

	while True:
		testConditions(test_img, max_size_ratio, max_curvature_ratio)
		countExpandableCC(test_img, max_distance_ratio)
		iterationCnt += 1
		print(iterationCnt)
		if expandable_CC == 0:
			break

	print("Done")
	CCs.clear()
	isBG = np.equal(test_img, [[BG_COLOR]*img.shape[1]] * img.shape[0])
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			if not isBG[i][j]:
				new_CC = CC()
				DFS(test_img, isBG, i, j, new_CC, len(CCs)+1)
				new_CC.resize()
				CCs.append(new_CC)

	print(len(CCs))

	# isBG = np.equal(test_img, [[BG_COLOR]*img.shape[1]] * img.shape[0])

	# bkimg = img.copy()
	# for i in range(img.shape[0]):
	# 	for j in range(img.shape[1]):
	# 		if isBG[i][j] == 1:
	# 			img[i][j] = BG_COLOR
	# 		else:
	# 			img[i][j] = FG_COLOR

	
	# cv2.imshow("xXx", np.hstack([bkimg, img]))
	# cv2.waitKey()
	word = "word"
	i = 0
	for cc in CCs:
		cv2.imwrite("word" + str(i) + ".png", img[cc.leftmost:cc.rightmost+1, cc.upmost:cc.downmost+1])
		i += 1
		# cv2.imshow("Char", img[cc.leftmost:cc.rightmost+1, cc.upmost:cc.downmost+1])
		# cv2.waitKey()

	


image = cv2.imread(args["image"])
quant = color_quantization(args, image)

gray_img = cv2.cvtColor(quant, cv2.COLOR_BGR2GRAY)
(thresh, bw_img) = cv2.threshold(gray_img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# kernel = np.ones((1, 2), np.uint8)
# new_img = cv2.erode(bw_img, kernel, iterations = 1)
# np.savetxt('tet.txt', bw_img, delimiter=',', fmt='%d')

text_strings(bw_img)
# cv2.imshow("image", np.hstack([bw_img]))
# cv2.waitKey(0)