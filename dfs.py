moves = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]

def DFS(isBG, _x, _y, im):
	res = [_y, _x, _y, _x]
	curNode = (_x, _y)
	stack = [curNode]
	nodes = [curNode]
	while stack:
		curNode = stack.pop()

		isBG[curNode[0]][curNode[1]] = True
		nodes.append((curNode[0],curNode[1]))
		res[0] = min(res[0], curNode[1])
		res[1] = min(res[1], curNode[0])
		res[2] = max(res[2], curNode[1])
		res[3] = max(res[3], curNode[0])

		for move in moves:
			x = curNode[0] + move[0]
			y = curNode[1] + move[1]
			if x >= 0 and x < isBG.shape[0] and y >= 0 and y < isBG.shape[1] \
			and not isBG[x][y]:
				stack.append((x, y))
	# if len(nodes) < im.shape[0]*im.shape[1]/1000:
	# 	for node in nodes:
	# 		im[node[0]][node[1]] = 255
	return res[0], res[1], res[2], res[3], nodes
