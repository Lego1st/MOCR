import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import cv2
import argparse
from getFeatures import getFeatures
from six.moves import cPickle as pickle

parser = argparse.ArgumentParser()
parser.add_argument("-root", help="root folder")
parser.add_argument("-data", help="data folder")
parser.add_argument("-size", help="image size")

args = parser.parse_args()

data_root = args.root # Change me to store data elsewhere 
def get_fols(root):
	l = os.listdir(root)
	l = [root + "/" + x for x in l if ".pickle" not in x]
	return l

def sort(l):
  l.sort(key=lambda x: int(x.split('/')[-1][1:]))

data_folders = get_fols(args.data)
sort(data_folders)
# print "train_folders: ", train_folders
# print "test_folders: ", test_folders
image_size = int(args.size)  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.

def load_letter(folder):
  """Load the data for a single letter label."""
  image_files = os.listdir(folder) 
  dataset = np.ndarray(shape=(len(image_files), 41), dtype=np.float32)
  print(folder)
  num_images = 0
  for image in image_files:
    image_file = os.path.join(folder, image)
    try:
      im = cv2.imread(image_file)
      if im.shape != (image_size, image_size, 3):
        raise Exception('%s\nUnexpected image shape: %s' % (image_file,str(im.shape)))
      features = getFeatures(im)
      dataset[num_images, :] = features
      num_images = num_images + 1
    except IOError as e:
      print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
    
  dataset = dataset[:num_images, :]
  print('Full dataset tensor:', dataset.shape)
  print('Mean:', np.mean(dataset))
  print('Standard deviation:', np.std(dataset))
  return dataset
        
def maybe_pickle(data_folders, force=False):
  dataset_names = []
  total = 0
  for folder in data_folders:
    set_filename = folder + '.pickle'
    dataset_names.append(set_filename)
    if os.path.exists(set_filename) and not force:
      # You may override by setting force=True.
      print('%s already present - Skipping pickling.' % set_filename)
      with open(set_filename, 'rb') as f:
        data = pickle.load(f)
        total += data.shape[0]
    else:
      print('Pickling %s.' % set_filename)
      dataset = load_letter(folder)
      total+= dataset.shape[0]
      try:
        with open(set_filename, 'wb') as f:
          pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
      except Exception as e:
        print('Unable to save data to', set_filename, ':', e)

  return dataset_names, total

train_datasets, total = maybe_pickle(data_folders)

print "train_datasets: ", train_datasets
print "%d examples" % total

def merge_datasets(pickle_files, totalSize):
  num_classes = len(pickle_files)
  DATASET = np.ndarray(shape=(totalSize, 41), dtype=np.float32)
  start= 0
  end= 0
  for label, pickle_file in enumerate(pickle_files):       
    try:
      with open(pickle_file, 'rb') as f:
        letter_set = pickle.load(f)
        end += letter_set.shape[0]
        DATASET[start:end, :] = letter_set
        start = end
    except Exception as e:
      print('Unable to process data from', pickle_file, ':', e)
      raise
    
  return DATASET
            
DATASET = merge_datasets(train_datasets, total)

print('Training:', DATASET.shape)

def randomize(dataset):
  permutation = np.random.permutation(dataset.shape[0])
  shuffled_dataset = dataset[permutation,:]
  return shuffled_dataset

DATASET = randomize(DATASET)

pickle_file = os.path.join(data_root, 'mocr_svm.pickle')

try:
  f = open(pickle_file, 'wb')
  save = {
    'train_dataset': DATASET,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise

statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)  