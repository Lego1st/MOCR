import cv2 
import numpy as np 
import color_quantization as CQ

im = cv2.imread('images/6.png')
print(CQ.quantize(2, im))