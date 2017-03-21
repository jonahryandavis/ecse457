''' Basic canny edge detection with some parameters that could be adjusted by 
machine learning '''

import cv2
import numpy as np 

def Canny(img)
	detected_edges = cv2.GaussianBlur(img, (),)
	cv2.Canny(img, detected_edges, lowThreshold, lowThreshold*ratio, apertureSize = ap_size, True)
