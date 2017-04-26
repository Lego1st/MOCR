import os
import numpy as py
import cv2
from matplotlib import pyplot as plt

tmp = cv2.imread('ImageFile1.png', 1)
h, w, ch = tmp.shape
sam = tmp[1 : h, 1 : w]

step = 30
numOfCharacters = 109
cwd = os.getcwd()
s = '.png'
countI = 0
for i in range(0, h - 1, step + 1):
    sI = 'c' + str(countI)
    cwdI = cwd + '//' + sI
    if (not os.path.exists(cwdI)):
        os.makedirs(cwdI)
        
    countJ = 245
    for j in range(0, w - 1, step + 1 ):
        piece = sam[i : i + step, j : j + step]
        sJ = '//' + str(countJ) + s
        cwdJ = cwdI + sJ
        cv2.imwrite(cwdJ, piece)
        countJ += 1
    countI += 1
