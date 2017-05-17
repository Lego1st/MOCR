import os
import sys
import numpy as np 
reload(sys)
sys.setdefaultencoding("utf-8")
data_root = '/home/tailongnguyen/MOCR_test/'
l = [1,2,3]
# np.save(data_root+'draft.npy', l)
n = np.load(data_root+'draft.npy')
print n