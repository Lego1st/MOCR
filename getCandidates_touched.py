import numpy as np 
def score_touched(x):
	"Calculate score of each column. High score increases the probability of being the boundary"
	sc = []
	for i, col in enumerate(x):
		# print v
		j, k = i, i
		pl, pr = 0, 0
		while j >= 1 and x[j] <= x[j-1]:
			j-= 1
			pass
		while k <= len(x)-2 and x[k] <= x[k+1]:
			# print i,k
			k+= 1
			pass
		pl = x[j] if j >= 0 else 0
		pr = x[k] if k <= len(x)-1 else 0
		# print pl, pr, col, max((float)(pl + pr - 2*col)/(col+1), 0) 
		sc.append(max((float)(pl + pr - 2*col)/(col+1), 0))

	"Find clusters of candidates"
	cnt = 0
	hill = []
	if sc[0] > 0:
		hill.append(0)
		cnt = 1
	for i, col in enumerate(sc):
		if i == 0:
			continue
		if cnt == 0:
			if sc[i] > 0 and sc[i-1] == 0:
				hill.append(i)
				cnt = 1
		if cnt == 1:
			if sc[i] == 0 and sc[i-1] > 0:
				hill.append(i-1)
				cnt = 0 
	if sc[len(sc)-1] > 0:
		hill.append(len(sc)-1)

	"Find local maximum in each cluster to reduce the number of candidates"
	left, right = 0,1
	localMax = []
	for t in range(len(hill)/2):
		MAX = -1
		temp = 0
		for i in range(hill[left], hill[right]+1):
			if sc[i] > MAX:
				MAX = sc[i]
				temp = i
		localMax.append(temp)
		left +=2
		right +=2

	candidates = []
	for i, c in enumerate(sc):
		if i in localMax:
			candidates.append(c)
		else:
			candidates.append(0)
	highest = max(candidates)
	for i, c in enumerate(candidates):
		if c < highest/10:
			candidates[i]=0
	# print candidates
	return sc, candidates, localMax
