import numpy as np 
def score_single(x):
	"Find column with no black pixel"
	candidates = []
	for i, col in enumerate(x):
		if col == 0 and x[i-1] != 0:
			candidates.append(i)
	return candidates
