import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from scipy import ndimage
from six.moves import cPickle as pickle
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-root", help="root folder")
parser.add_argument("-train_fol", help="train folder")
parser.add_argument("-test_fol", help="test folder")
parser.add_argument("-size", help="image size")
parser.add_argument("-channel", help="image channels")
parser.add_argument("-train", help="train size")
parser.add_argument("-valid", help="valid size")
parser.add_argument("-test", help="test size")
parser.add_argument("-min_train", help="min number of training per class")
parser.add_argument("-min_test", help="min number of testing per class")


args = parser.parse_args()

data_root = args.root # Change me to store data elsewhere 
def get_fols(root):
	l = os.listdir(root)
	l = [root + "/" + x for x in l]
	return l

train_folders = get_fols(args.train_fol)
test_folders =get_fols(args.test_fol)

print "train_folders: ", train_folders
print "test_folders: ", test_folders
image_size = int(args.size)  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.
channel = int(args.channel)
def load_letter(folder, min_num_images):
  """Load the data for a single letter label."""
  image_files = os.listdir(folder)
  dataset = np.ndarray(shape=(len(image_files), image_size, image_size, channel),
                         dtype=np.float32)
  print(folder)
  num_images = 0
  for image in image_files:
    image_file = os.path.join(folder, image)
    try:
      image_data = (ndimage.imread(image_file).astype(float) - 
                    pixel_depth / 2) / pixel_depth
      if image_data.shape != (image_size, image_size, channel):
        raise Exception('Unexpected image shape: %s' % str(image_data.shape))
      dataset[num_images, :, :, :] = image_data
      num_images = num_images + 1
    except IOError as e:
      print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
    
  dataset = dataset[0:num_images, :, :, :]
  if num_images < min_num_images:
    raise Exception('Many fewer images than expected: %d < %d' %
                    (num_images, min_num_images))
    
  print('Full dataset tensor:', dataset.shape)
  print('Mean:', np.mean(dataset))
  print('Standard deviation:', np.std(dataset))
  return dataset
        
def maybe_pickle(data_folders, min_num_images_per_class, force=False):
  dataset_names = []
  for folder in data_folders:
    set_filename = folder + '.pickle'
    dataset_names.append(set_filename)
    if os.path.exists(set_filename) and not force:
      # You may override by setting force=True.
      print('%s already present - Skipping pickling.' % set_filename)
    else:
      print('Pickling %s.' % set_filename)
      dataset = load_letter(folder, min_num_images_per_class)
      try:
        with open(set_filename, 'wb') as f:
          pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
      except Exception as e:
        print('Unable to save data to', set_filename, ':', e)
  
  return dataset_names

train_datasets = maybe_pickle(train_folders, int(args.min_train))
test_datasets = maybe_pickle(test_folders, int(args.min_test))

print "train_datasets: ", train_datasets
print "test_datasets: ", test_datasets

def make_arrays(nb_rows, img_size, channel):
  if nb_rows:
    dataset = np.ndarray((nb_rows, img_size, img_size, channel), dtype=np.float32)
    labels = np.ndarray(nb_rows, dtype=np.int32)
  else:
    dataset, labels = None, None
  return dataset, labels

def merge_datasets(pickle_files, train_size, valid_size=0):
  num_classes = len(pickle_files)
  train_dataset, train_labels = make_arrays(train_size, image_size, channel)
  valid_dataset, valid_labels = make_arrays(valid_size, image_size, channel)
  tsize_per_class = train_size // num_classes
  vsize_per_class = valid_size // num_classes
    
  start_v, start_t = 0, 0
  end_v, end_t = vsize_per_class, tsize_per_class
  end_l = vsize_per_class+tsize_per_class
  for label, pickle_file in enumerate(pickle_files):       
    try:
      with open(pickle_file, 'rb') as f:
        letter_set = pickle.load(f)
        # let's shuffle the letters to have random validation and training set
        np.random.shuffle(letter_set)
        if valid_dataset is not None:
          valid_letter = letter_set[:vsize_per_class, :, :, :]
          valid_dataset[start_v:end_v, :, :, :] = valid_letter
          valid_labels[start_v:end_v] = label
          start_v += vsize_per_class
          end_v += vsize_per_class
                    
        train_letter = letter_set[vsize_per_class:end_l, :, :]
        train_dataset[start_t:end_t, :, :] = train_letter
        train_labels[start_t:end_t] = label
        start_t += tsize_per_class
        end_t += tsize_per_class
    except Exception as e:
      print('Unable to process data from', pickle_file, ':', e)
      raise
    
  return valid_dataset, valid_labels, train_dataset, train_labels
            
            
train_size = int(args.train)
valid_size = int(args.valid)
test_size = int(args.test)

valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(
  train_datasets, train_size, valid_size)
_, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)

print('Training:', train_dataset.shape, train_labels.shape)
if valid_size != 0:
	print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)
def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels
train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
if valid_size != 0:
	valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)
pickle_file = os.path.join(data_root, 'mocr.pickle')

try:
  f = open(pickle_file, 'wb')
  save = {
    'train_dataset': train_dataset,
    'train_labels': train_labels,
    'valid_dataset': valid_dataset,
    'valid_labels': valid_labels,
    'test_dataset': test_dataset,
    'test_labels': test_labels,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise

statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)  