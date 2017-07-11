import numpy as np 
import cv2
import argparse
from dfs import *

# parser = argparse.ArgumentParser()
# parser.add_argument("-i", "--image", required = True, help = "Path to the image")
# args = vars(parser.parse_args())

# im = cv2.imread(args['image'])
# gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
# retVal, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
MIN_PIXELS = 10
# SPACE_SIZE = 30

def get_CCs(image):
	# cv2.imshow('word', image)
	# cv2.waitKey()
	CCs = []
	CCs_id = np.asarray([[-1]*image.shape[1]]*image.shape[0])
	# print "BEFORE: ", CCs_id, CCs_id.shape
	isBG = np.equal(image, [[255]*image.shape[1]]*image.shape[0])
	idx = 0
	for i in range(image.shape[0]):
		for j in range(image.shape[1]):
			if not isBG[i][j]:
				temp = DFS(isBG, i, j, image)
				if len(temp[-1]) > MIN_PIXELS:
					CCs.append(temp)
					for p in temp[-1]:
						CCs_id[p[0]][p[1]] = idx
					idx +=1
					# print CCs_id
	return CCs, CCs_id

def expandable(pixel, top, bot):
	if pixel[0] - top < (bot-top+1)/2:
		return True
	# print "Don't expand ", pixel
	return False


def expand_CC(image, CC, CCs_id, CCs):
	left, top, right, bot = CC[:4]
	pixels = CC[-1]
	# cv2.imshow('cc_in_pre', image)
	# cv2.waitKey()
	# img = image.copy()
	# for p in pixels:
	# 	img[p[0]][p[1]] = 150
	# cv2.imshow('cc_in', img)
	# cv2.waitKey()
	# print CC[:4]
	
	for p in pixels:
		if not expandable(p, top, bot):
			continue
		# expand to right
		for i in range(p[1]+1, image.shape[1]):
			if image[p[0]][i] != 0 and CCs_id[max(p[0]-1, 0)][i] == -1:
				if i < image.shape[1]-1:
					image[p[0]][i] = 0
				else:
					for j in range(p[1]+1, image.shape[1]):
						image[p[0]][j] = 255
			else:
				if image[p[0]][i] == 0:
					ID = CCs_id[p[0]][i]
				if CCs_id[max(p[0]-1, 0)][i] != -1:
					ID = CCs_id[max(p[0]-1, 0)][i]
				if ID == -1:
					break
				current_height = bot-top+1
				height = CCs[ID][3] - CCs[ID][1]+1
				overlay = min(CCs[ID][3],bot) - max(CCs[ID][1], top)
				if height/current_height > 2 or ((float)(overlay)/current_height < 0.5 and (float)(overlay)/height < 0.5):
					for j in range(p[1]+1, i):
						image[p[0]][j] = 255
				break
		# expand to left
		for i in range(p[1]-1, -1, -1):
			if image[p[0]][i] != 0 and CCs_id[max(p[0]-1, 0)][i] == -1:
				if i > 0:
					image[p[0]][i] = 0
				else:
					for j in range(p[1]):
						image[p[0]][j] = 255
			else:
				if image[p[0]][i] == 0:
					ID = CCs_id[p[0]][i]
				if CCs_id[max(p[0]-1, 0)][i] != -1:
					ID = CCs_id[max(p[0]-1, 0)][i]
				if ID == -1:
					break
				current_height = bot-top+1
				height = CCs[ID][3] - CCs[ID][1]+1
				overlay = min(CCs[ID][3],bot) - max(CCs[ID][1], top)
				if height/current_height > 2 or ((float)(overlay)/current_height < 0.5 and (float)(overlay)/height < 0.5) :
					for j in range(i+1, p[1]):
						image[p[0]][j] = 255
				break

def join(cc1, cc2):
	# temp = cc1[-1] + cc2[-1]
	# print "CC1:" 
	# print cc1[-1] + cc2[-1]
	return min(cc1[0], cc2[0]), min(cc1[1], cc2[1]), max(cc1[2], cc2[2]), max(cc1[3], cc2[3]), cc1[-1] + cc2[-1]

def get_draft(cc, CCs_id):
	temp = np.asarray([[255]*(cc[2] - cc[0]+1)]*(cc[3]-cc[1]+1), dtype = np.uint8)
	for p in cc[-1]:
		if CCs_id[p[0]][p[1]] != -1:
			temp[p[0]-cc[1]][p[1]-cc[0]] = 0
	return temp
# extract_lines(thresh)

def extract_lines(image):
	lines = []
	average_cc_height = []
	img = image.copy()
	# print img.shape
	# cv2.imshow('before', img)
	# cv2.waitKey()
	CCs, CCs_id = get_CCs(img)
	# print CCs_id
	for cc in CCs:
		average_cc_height.append(cc[3]-cc[1]+1)
	average_cc_height = (float)(sum(average_cc_height))/len(average_cc_height)
	print average_cc_height
	for i, cc in enumerate(CCs):
		if cc[3]-cc[1] > average_cc_height/2:
			expand_CC(img, cc, CCs_id, CCs)
		# cv2.imshow('after cc', img)
		# cv2.waitKey()
	new_CCs, new_CCs_id = get_CCs(img)
	# for i, cc in enumerate(new_CCs):
	# 	img1 = img.copy()
	# 	print cc[3]-cc[1] 
	# 	print cc[2]-cc[0] 
	# 	for p in cc[-1]:
	# 		img1[p[0]][p[1]] = 150
	# 		ID = new_CCs_id[p[0]][p[1]]
	# 	cv2.imshow(str(ID) + "_" + str(i), img1)	
	# 	cv2.waitKey()
	del_cc = []
	for t, cc in enumerate(new_CCs):
		if cc[3]-cc[1] < 10 or cc[2]-cc[0] < 10:
			# if cc[2] - cc[0] > 20 and cc[3]-cc[1] < 10:
			# 	del_cc.append(t)
			# expand to nearest cc
			# img1 = img.copy()
			# for p in cc[-1]:
			# 	img1[p[0]][p[1]] = 150
			# cv2.imshow("joining", img1)	
			# cv2.waitKey()

			pixels = cc[-1]
			pixels.sort(key=lambda x: x[0])
			p1 = pixels[0]
			p2 = pixels[-1]
			i = p1[0]-1
			j = p2[0]+1
	
			hh = new_CCs_id.shape[0]
			if 0 <= i < hh and 0 <= j < hh:
				while new_CCs_id[i][p1[1]] == -1 and new_CCs_id[j][p2[1]] == -1:
					i-=1
					j+=1
					if (i == max(0, p1[1] - average_cc_height/2) and j == min(img.shape[0]-1, p2[1] + average_cc_height/2)):
						break
					if not (0 <= i < hh and 0 <= j < hh):
						break
			if not (0 <= i < hh and 0 <= j < hh):
				continue
			if new_CCs_id[i][p1[1]] != -1:
				ID = new_CCs_id[i][p1[1]]
				# print "touch ID: ", ID
				# img1 = img.copy()
				# for p in new_CCs[ID][-1]:
				# 	img1[p[0]][p[1]] = 150
				# cv2.imshow("before join", img1)	
				# cv2.waitKey()
				new_CCs[ID] = list(join(new_CCs[ID], cc))
				# img1 = img.copy()
				# for p in new_CCs[ID][-1]:
				# 	img1[p[0]][p[1]] = 150
				# cv2.imshow("after join", img1)	
				# cv2.waitKey()
			elif new_CCs_id[j][p2[1]] != -1:
				ID = new_CCs_id[j][p2[1]]
				# print "touch ID: ", ID
				# img1 = img.copy()
				# for p in new_CCs[ID][-1]:
				# 	img1[p[0]][p[1]] = 150
				# cv2.imshow("before join", img1)	
				# cv2.waitKey()
				new_CCs[ID] = list(join(new_CCs[ID], cc))
				# img1 = img.copy()
				# for p in new_CCs[ID][-1]:
				# 	img1[p[0]][p[1]] = 150
				# cv2.imshow("after join", img1)	
				# cv2.waitKey()
			del_cc.append(t)

	for i, cc in enumerate(new_CCs):
		if i in del_cc:
			continue
		# img1 = img.copy()
		# print cc[-1]
		# for p in cc[-1]:
		# 	img1[p[0]][p[1]] = 150
		# cv2.imshow(str(t), img1)	
		# cv2.waitKey()
		
		# draft = get_draft(cc, CCs_id)
		lines.append(get_draft(cc, CCs_id))
		# cv2.imshow('draft' + str(t), draft)	
		# cv2.waitKey()


	# cv2.imshow('test', img)
	# cv2.waitKey()
	return lines



		
		





