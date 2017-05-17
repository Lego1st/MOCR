import numpy as np
import os
import sys
from scipy import ndimage
import pickle
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-root", help="root folder")
parser.add_argument("-data", help="data folder")
parser.add_argument("-size", help="image size")

args = parser.parse_args()
reload(sys)
sys.setdefaultencoding("utf-8")
data_root = args.root # Change me to store data elsewhere 
def get_fols(root):
	l = os.listdir(root)
	l = [root + "/" + x for x in l if ".pickle" not in x]
	return l

def sort(l):
  l.sort(key=lambda x: x.split('/')[-1])

train_folders = get_fols(args.data)
sort(train_folders)
print( "train_folders: ", train_folders)
image_size = int(args.size)  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.
total = 0

def load_letter(folder):
  """Load the data for a single letter label."""
  image_files = os.listdir(folder) 
  dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32)
  print(folder)
  num_images = 0
  for image in image_files:
    image_file = os.path.join(folder, image)
    try:
      image_data = (ndimage.imread(image_file).astype(float) - 
                    pixel_depth / 2) / pixel_depth
      if image_data.shape != (image_size, image_size):
        raise Exception('%s\nUnexpected image shape: %s' % (image_file,str(image_data.shape)))
      dataset[num_images, :, :] = image_data
      num_images = num_images + 1
    except IOError as e:
      print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
    
  dataset = dataset[0:num_images, :, :]

  print('Full dataset tensor:', dataset.shape)
  print('Mean:', np.mean(dataset))
  print('Standard deviation:', np.std(dataset))
  return dataset
        
def maybe_pickle(data_folders, force=False):
  dataset_names = []
  for folder in data_folders:
    set_filename = folder + '.pickle'
    dataset_names.append(set_filename)
    if os.path.exists(set_filename) and not force:
      # You may override by setting force=True.
      print('%s already present - Skipping pickling.' % set_filename)
      # with open(set_filename, 'rb') as f:
        # dataset = pickle.load(f)
        # global total
        # total += dataset.shape[0]
    else:
      print('Pickling %s.' % set_filename)
      dataset = load_letter(folder)
      try:
        with open(set_filename, 'wb') as f:
          pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
      except Exception as e:
        print('Unable to save data to', set_filename, ':', e)
  
  return dataset_names

train_datasets = maybe_pickle(train_folders)

# print "train_datasets: ", train_datasets

def make_arrays(nb_rows, img_size):
  if nb_rows:
    dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
    labels = np.ndarray(nb_rows, dtype=np.int32)
  else:
    dataset, labels = None, None
  return dataset, labels
total = 2000000
def merge_datasets(pickle_files, number_of_samples):
  num_classes = len(pickle_files)
  train_dataset, train_labels = make_arrays(number_of_samples, image_size)
  valid_dataset, valid_labels = make_arrays(number_of_samples, image_size)
  test_dataset, test_labels = make_arrays(number_of_samples, image_size)
  start_t, end_t = 0, 0
  start_v, end_v = 0, 0
  start_test, end_test = 0, 0
  t = 0
  for label, pickle_file in enumerate(pickle_files):       
    try:
      with open(pickle_file, 'rb') as f:
        letter_set = pickle.load(f)
        # let's shuffle the letters to have random validation and training set
        np.random.shuffle(letter_set)                    
        l = letter_set.shape[0]
        t += l 
        end_t += l*6/10
        end_v += l*2/10
        end_test += l - l*6/10 - l*2/10
        print( "Size: ", t)
        print( "Length: ", l)
        print( start_t, end_t)
        print( start_v, end_v)
        print( start_test, end_test)
        train_dataset[start_t:end_t, :, :] = letter_set[0:l*6/10]
        train_labels[start_t:end_t] = label
        valid_dataset[start_v:end_v, :, :] = letter_set[l*6/10:l*6/10+l*2/10]
        valid_labels[start_v:end_v] = label
        test_dataset[start_test:end_test, :, :] = letter_set[l*6/10+l*2/10:]
        test_labels[start_test:end_test] = label
        start_t = end_t
        start_v = end_v
        start_test = end_test
    except Exception as e:
      print('Unable to process data from', pickle_file, ':', e)
      raise
    
  return train_dataset[:end_t,:,:] , train_labels[:end_t] , valid_dataset[:end_v, :, :], valid_labels[:end_v], test_dataset[:end_test, :, :], test_labels[:end_test]
            
train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels= merge_datasets(
  train_datasets, total)

print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)

def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels

train_dataset, train_labels = randomize(train_dataset, train_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)


try:
  pickle_train_file = os.path.join(data_root, 'train.npy')
  pickle_valid_file = os.path.join(data_root, 'valid.npy')
  pickle_test_file = os.path.join(data_root, 'test.npy')
  pickle_train_label = os.path.join(data_root, 'train_labels.npy')
  pickle_valid_label = os.path.join(data_root, 'valid_labels.npy')
  pickle_test_label = os.path.join(data_root, 'test_labels.npy')
  np.save(pickle_train_file, train_dataset)
  np.save(pickle_valid_file, valid_dataset)
  np.save(pickle_test_file, test_dataset)
  np.save(pickle_train_label, train_labels)
  np.save(pickle_valid_label, valid_labels)
  np.save(pickle_test_label, test_labels)
except Exception as e:
  print('Unable to save data to:', e)
  raise


# try:
#   pickle_test_file = os.path.join(data_root, 'mocr_test.npy')
#   f = open(pickle_test_file, 'wb')
#   save = {
#     'test_dataset': test_dataset,
#     'test_labels': test_labels,
#     }
#   pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
#   f.close()
# except Exception as e:
#   print('Unable to save data to', pickle_test_file, ':', e)
#   raise

# statinfo = os.stat(pickle_file)
# print('Compressed pickle size:', statinfo.st_size)  