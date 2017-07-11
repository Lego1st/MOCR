from sklearn.cluster import KMeans
import numpy as np 

X = np.array([2,3,10,2,4,12,3,2,4,11,4,1,3,13,2,2,3,12])
for i in range(X.shape[0]):
	X[i] = [X[i], 0]
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
print kmeans.labels_