import numpy as np 

def project(im):
	x = []
	for i in range(im.shape[1]):
		count = 0
		for j in range(im.shape[0]):
			if np.array_equal(im[j][i],[0, 0, 0]):
				count +=1
		x.append(count)
	return x

