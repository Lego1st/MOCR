# import the necessary packages
import argparse
import cv2
import numpy as np
# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
cropping = False

def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
	global refPt, cropping
 
	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
	if event == cv2.EVENT_LBUTTONDOWN:
		refPt = [(x, y)]
		cropping = True
 
	# check to see if the left mouse button was released
	elif event == cv2.EVENT_LBUTTONUP:
		# record the ending (x, y) coordinates and indicate that
		# the cropping operation is finished
		refPt.append((x, y))
		cropping = False
 
		# draw a rectangle around the region of interest
		cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 1)
		cv2.imshow(folder, image)

def rot_img(img, deg):
	rows, cols = img.shape[:2]
	M = cv2.getRotationMatrix2D((cols/2,rows/2),deg,1)
	return cv2.warpAffine(img, M, (cols,rows))

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True, help="Path to the image")
# args = vars(ap.parse_args())
 
# # load the image, clone it, and setup the mouse callback function
# image = cv2.imread(args["image"])

idx = 0
image = np.array([])
folder = "image"

def select(img, fldr):
	global image
	global idx
	global folder
	folder = fldr
	image = img.copy() 
	clone = image.copy()
	cv2.namedWindow(folder)
	cv2.setMouseCallback(folder, click_and_crop)
	rot = 0
	prev_rot = 0
	idx = 0
	# keep looping until the 'q' key is pressed

	while True:
		# display the image and wait for a keypress
		if rot != prev_rot:
			# print(rot)
			prev_rot = rot
			image = rot_img(clone.copy(), rot)
		cv2.imshow(folder, image)
		key = cv2.waitKey(1) & 0xFF
	 
		# if the 'r' key is pressed, reset the cropping region
		if key == ord("r"):
			image = rot_img(clone.copy(), rot)
			refPt.clear()
	 
		# if the 'c' key is pressed, break from the loop
		elif key == ord("c"):
			if len(refPt) == 2:
				image = rot_img(clone.copy(), rot)
				roi = image[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
				# cv2.imshow("ROI", roi)
				# cv2.waitKey()
				cv2.imwrite(folder + "/" + str(idx) + ".png", roi)
				idx += 1

		elif key == ord("a"):
			rot += 1

		elif key == ord("d"):
			rot -= 1

		elif key == ord("q"):
			break
	 
	# if there are two reference points, then crop the region of interest
	# from teh image and display it

	 
	# close all open windows
	cv2.destroyAllWindows()