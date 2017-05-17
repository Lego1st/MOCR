import cv2
import color_quantization as CQ 
import numpy as np 

moves = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]

def DFS(isBG, _x, _y):
	res = [_y, _x, _y, _x]
	curNode = (_x, _y)
	stack = [curNode]
	while stack:
		curNode = stack.pop()

		isBG[curNode[0]][curNode[1]] = True
		res[0] = min(res[0], curNode[1])
		res[1] = min(res[1], curNode[0])
		res[2] = max(res[2], curNode[1])
		res[3] = max(res[3], curNode[0])

		for move in moves:
			x = curNode[0] + move[0]
			y = curNode[1] + move[1]
			if x >= 0 and x < isBG.shape[0] and y >= 0 and y < isBG.shape[1] \
			and not isBG[x][y]:
				stack.append((x, y))

	return res[0], res[1], res[2], res[3]

def findLines(image):
	res = []
	# img = image.copy()
	isBG = np.equal(image, [[255]*image.shape[1]]*image.shape[0])
	for i in range(image.shape[0]):
		for j in range(image.shape[1]):
			if not isBG[i][j]:
				res.append(DFS(isBG, i, j))
	# print("result: ", res)

	return res

def segmentize(thresh, space_size = 15):
	# img = cv2.imread('images/book2.png')

	# img = CQ.quantize(2, img)

	# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		
	# ret,thresh = cv2.threshold(gray_img,127,255,0)

	bin_img = thresh.copy()

	for i in range(thresh.shape[0]):
		for j in range(thresh.shape[1]):
				bin_img[i][j] = (1 if thresh[i][j] == 0 else 0)

	kernel = np.ones((1, space_size))

	dilation = cv2.dilate(bin_img, kernel, iterations = 1)

	# cv2.imshow("dialtion", 255*(1-dilation))
	# cv2.waitKey()

	# _, contours, _ = cv2.findContours(255*(1-dilation),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)	
	contours = findLines(dilation)	
	res = []

	for i in range(0, len(contours)):
	       # cnt = contours[i]
	       # x,y,w,h = cv2.boundingRect(cnt)
	       # if h < 0 or w < 0 or w < 100: 
	       # 	continue
	       # cv2.rectangle(thresh,(x,y),(x+w,y+h),(0,255,0),2)
	       # res.append((x, y, x+w, y +h))
	       # cv2.imshow("xx", img[y: y+h, x: x+w])
	       # cv2.waitKey()

	       x, y, u, v = contours[i]
	       if u < x or v < y or u-x < 10 or v-y < 10:
	       	continue
	       # cv2.rectangle(thresh,(x, y), (u, v), (0, 255, 0), 2)
	       res.append(contours[i])

	# cv2.imshow("xx", thresh)
	# cv2.waitKey()
	return res