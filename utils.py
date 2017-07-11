# -*- coding: UTF-8 -*-
from scipy import ndimage
from scipy.misc import imresize
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import numpy as np 
import os
import pickle 

np.random.seed(50)
# s = "0123456789abcdefghijklmnopqrstuvwxyzàáâãèéêìíòóôõùúýăđĩũơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ "
s = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz "
forward_mapping = {}
backward_mapping = {}
for i, c in enumerate(unicode(s, 'utf-8')):
	forward_mapping[c] = i+1
	backward_mapping[i+1] = c

def pyramid(image, scale=1.5, minSize=(30, 30)):
	# yield the original image
	yield image
	# keep looping over the pyramid
	while True:
		# compute the new dimensions of the image and resize it
		w = int(image.shape[1] / scale)
		h = int(image.shape[0] / scale)
		image = cv2.resize(image,(w,h))
		# if the resized image does not meet the supplied minimum
		# size, then stop constructing the pyramid
		if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
			break
		# yield the next image in the pyramid
		yield image

def normalize(image):
	return (image.astype(float)- 255.0/2) / 255.0

def sliding_window(image, stepSize):
	# slide a window across the image
	num = 0
	for x in xrange(0, image.shape[1], stepSize):
		# yield the current window
		if x + image.shape[0] <= image.shape[1] or num == 0:
			num += 1
			yield normalize(imresize(image[:, x:x + image.shape[0]], (32,32)))

def text_to_labels(text, mapping=forward_mapping):
    return [mapping[char] for char in unicode(text,'utf-8').lower()]

def labels_to_text(labels, mapping=backward_mapping):
	ret = [mapping[l] for l in labels if l != 0]
	return ''.join(ret)
	# return ret

def to_features_vecs(items, model):
    return np.asarray([model.features_vec([item, True])[0] for item in items])

def get_batches(dataset, batch_size = 1):
	np.random.shuffle(dataset)
	samples, labels = zip(*dataset)
	# print dataset
	idx = 0
	while idx < len(dataset):
		print samples[idx: idx+batch_size]
		patches = [ [window for window in sliding_window(ndimage.imread(link), 10)] for link in samples[idx : idx + batch_size] ] 
		yield [ np.asarray(pad_sequences(patches, padding='post', value=255)), \
				np.concatenate(labels[idx : idx + batch_size]), \
				np.asarray([len(p) for p in patches]), \
				np.asarray([len(l) for l in labels[idx : idx + batch_size]]) ]
		idx += batch_size 


def reshape(im):
	return im.reshape(im.shape[0], im.shape[1], 1)
def get_batches_crnn(dataset, batch_size=1):
	np.random.shuffle(dataset)
	samples, labels = zip(*dataset)
	idx = 0
	while idx < len(dataset):
		patches = [ np.transpose(reshape(ndimage.imread(link)), (1,0,2))  for link in samples[idx : idx + batch_size] ]
		yield [ np.asarray(patches), \
				patches[0].shape[0],
				np.concatenate(labels[idx : idx + batch_size]), \
				np.asarray([p.shape[0]//4 for p in patches]), \
				np.asarray([len(l) for l in labels[idx : idx + batch_size]]) ]
		idx += batch_size 



# for b in get_batches(zip([samples[2]], [labels[2]]), 1):
# 	print b[0].shape, b[1], b[2], b[3]

# s = "0123456789abcdefghijklmnopqrstuvwxyzàáâãèéêìíòóôõùúýăđĩũơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ "
# mapping = {}
# d = {}
# for i, c in enumerate(unicode(s, 'utf-8')):
# 	mapping[c] = i
# 	d[i] = c

# root = '/home/tailongnguyen/aeh16/train'
# words = os.listdir(root)
# words_dir = [os.path.join(root, f) for f in words]
# samples = []
# for w in words_dir:
#     samples.extend([os.path.join(w, f) for f in os.listdir(w)])

# labels = []
# for i, w in enumerate(words):
#     labels.extend([w]* len(os.listdir(words_dir[i])))
# labels = [text_to_labels(l, forward_mapping) for l in labels]

# r = np.random.randint(0, len(samples))
# print samples[r], labels[r]
# for b in get_batches_crnn(zip([samples[r]], [labels[r]]), 1):
# 	# p = M.predict_step([b[0], b[1], b[3], True])[0]
# 	print b[0].shape
	# print b[2]
# print "True: ", labels_to_text(labels[r])
# print "Predict: ", labels_to_text(p[0])
# print labels[:4]

# for l in labels[:4]:
# 	print labels_to_text(l)