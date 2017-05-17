import numpy as np
import pickle
import cv2
import argparse
from sklearn import svm
from sklearn.externals import joblib
	
X_train = pickle.load(open('mocr_svm.pickle', 'rb'))['train_dataset']
# fit the model
# clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
# clf.fit(X_train)
# joblib.dump(clf, 'svm.pkl') 
clf = joblib.load('svm.pkl')
# y_pred_train = clf.predict(X_train)
# y_pred_test = clf.predict(X_test)
# y_pred_outliers = clf.predict(X_outliers)
# n_error_train = y_pred_train[y_pred_train == -1].size
# print "Error: %d/%d" % (n_error_train, X_train.shape[0])
# n_error_test = y_pred_test[y_pred_test == -1].size
# n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size

# score_train = clf.decision_function(X_train)
# print score_train
# score_test  = clf.decision_function(X_test)
# score_outlier = clf.decision_function(X_outliers)
