""" Basic canny edge detection with some parameters that could be adjusted by 
machine learning """

import cv2
import os
import sys 
import numpy as np 
from matplotlib import pyplot as plt

def canny(img, lowThreshold, ratio, ap_size):
	#use canny filter with passed parameters 
	detected_edges = cv2.Canny(img, lowThreshold, lowThreshold*ratio)
	return detected_edges

def main(): 
	img = cv2.imread('20_masked.jpg')
	detected_edges = canny(img, 1, 200, 3)
	cv2.imshow('img', detected_edges)                                   # Display
	cv2.waitKey(0)
	detected_edges_stack = np.dstack([detected_edges]*3)    # Create 3-channel alpha mask
	#-- Blend masked img into MASK_COLOR background --------------------------------------
	detected_edges_stack  = detected_edges_stack.astype('float32') / 255.0          # Use float matrices
	detected_edges_stack = (detected_edges_stack * 255).astype('uint8')             # Convert back to 8-bit 
	img_out = cv2.cvtColor(detected_edges_stack,cv2.COLOR_BGR2GRAY)  # Convert back to 8-bit 

	cv2.imshow('img', img_out)                                   # Display
	cv2.waitKey(0)
	cv2.imwrite('20_canny.jpg',img_out)  
	return 0	

if __name__ == "__main__":
	sys.exit(main())

