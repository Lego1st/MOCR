import numpy as np 
import cv2
moves = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]
import argparse
import cv2

def DFS(isBG, _x, _y, im):
	res = [_y, _x, _y, _x]
	curNode = (_x, _y)
	stack = [curNode]
	nodes = [curNode]
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
				nodes.append((x,y))
	if len(nodes) < im.shape[0]*im.shape[1]/1000:
		for node in nodes:
			im[node[0]][node[1]] = 255
	return res[0], res[1], res[2], res[3], nodes

def find_characters(image):
	# cv2.imshow('word', image)
	# cv2.waitKey()
	res = []
	chars = []
	isBG = np.equal(image, [[255]*image.shape[1]]*image.shape[0])

	for i in range(image.shape[0]):
		for j in range(image.shape[1]):
			if not isBG[i][j]:
				temp = DFS(isBG, i, j, image)
				if len(temp[-1]) > image.shape[0]*image.shape[1]/1000:
					res.append(temp)

	res.sort(key=lambda x: x[0])
	res.append((10000,10000,10000,[]))
	i = 0

	while i < len(res)-1:
		# img1 = cv2.rectangle(image, (res[i][0],res[i][1]), (res[i][2],res[i][3]), (0,255,0), 1)
		# cv2.imshow('hollywood', img1)
		# cv2.waitKey()
		catch = []
		if res[i+1][0] <= res[i][2]:
			if res[i+1][3] <= res[i][1]:
				# print "Catch ", i, i+1
				top = res[i][1]
				bot = res[i][3]
				right = res[i][2]
				top = min(top, res[i+1][1])
				bot = max(bot, res[i+1][3])
				right = max(right, res[i+1][2])
				# print "Top: %d , Bottom: %d" % (top, bot)
				first = i
				last = i+1
				catch.extend(res[i][-1])
				catch.extend(res[i+1][-1])
				i+=2
				while res[i][0] <= res[first][2] and res[i][3] <= res[first][1]:
					# print "Dau at ", i
					top = min(top, res[i][1])
					bot = max(bot, res[i][3])
					# print "Top: %d , Bottom: %d" % (top, bot)
					last = i
					catch.extend(res[i][-1])
					i+=1
				# print "Fist: ", res[first][:4], " Last: ", res[last][:4]

				img = image[top:bot+1, res[first][0] : right+1].copy()
				for idx in range(img.shape[0]):
					for j in range(img.shape[1]):
						img[idx][j] = 255
				# print "Height: %d , Width: %d" % (img.shape)
				for pixel in catch:
					img[pixel[0]-top][pixel[1]-res[first][0]] = 0
					image[pixel[0]][pixel[1]] = 255

				chars.append(img)
				# cv2.imshow('Dau', img)
				# cv2.waitKey()
			else:
				img = image[res[i][1]:res[i][3]+1, res[i][0]: res[i][2]+1].copy()
				for idx in range(img.shape[0]):
					for j in range(img.shape[1]):
						img[idx][j] = 255
				# cv2.imshow('pre_touched', img)
				# cv2.waitKey()
				for pixel in res[i][-1]:
					# print "Painting at: ", pixel
					img[pixel[0]-res[i][1]][pixel[1]-res[i][0]] = 0
					image[pixel[0]][pixel[1]] = 255

				chars.append(img)
				i+=1
				# cv2.imshow('touched' + str(i), img)
				# cv2.waitKey()
		else:
			img = image[res[i][1]:res[i][3]+1, res[i][0]: res[i][2]+1].copy()
			for pixel in res[i][-1]:
					image[pixel[0]][pixel[1]] = 255

			chars.append(img)
			i +=1
			# cv2.imshow('single' + str(i), img)
			# cv2.waitKey()

	chars = [cv2.resize(c, (30,30)) for c in chars if c.shape[0]*c.shape[1] > image.shape[0]**2*0.05]
	return chars

# parser = argparse.ArgumentParser()
# parser.add_argument("-i", help="image path")
# args = parser.parse_args()
# im = cv2.imread(args.i)
# im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
# cv2.imshow('hollywood', im)
# cv2.waitKey()
# chars = find_characters(im)
# # print res
# for cnt in res:
# 	print cnt
# 	image = cv2.rectangle(image, (cnt[0],cnt[1]), (cnt[2],cnt[3]), (0,255,0), 1)
# cv2.imshow('hollywood', image)
# cv2.waitKey()
