import math
import Image 
import cv2
import sys 
import numpy as np

''' Compares the Canny generated edge map to the training set edge map image using the following parameters imgA is part of the training image, imgB is part of the canny edge detection generated image'''

class ImCompare(object):

	_pixel = 255

def mse(imA, imB):
# Computes the Mean Square Error between the two edge images. 
# mse is the sum of the squared difference between the two edge images
# both images must have the same dimensions
	err = np.sum((imA.astype("float") - imB.astype("float"))**2)
	err /= float(imA.shape[0]*imB.shape[1])
#returns the MSE, the lower the value the more similar 
	return err
	
def dice(imA, imB):
#Computes the dice value between the two edge images,  
	a = 0 #number of pixels of value '1' in both binary images 
	b = 0 #number of pixels taking value '1' in image B only
	c = 0 #number of pixels taking value '1' in image A only
	d = 0 #number of pixels taking value '0' in both binary images 
	_pixel = 255
	(n,m) = imA.shape
	for i in range(0,n):
		for j in range(0,m):
			if(imB[i][j] != 0):
				print(imB[i][j])
			if(imA[i][j] == _pixel): 
				if(imB[i][j] == _pixel):
					a = a+1
				else:
					c = c+1
			else:
				if(imB[i][j] == 0):
					d = d+1
				else:
					b = b+1
	#if images overlap perfectly then Dice = 1, if not overlap then Dice = 0
	dice = (2*a)/(2*a + b - c)
	return dice

def ssim(imA, imB):
	_pixel = 255
#computes structural similarity index between two images
	if(_pixel == 1):
		L = 0
	else:	
		L = _pixel 
	K1 = 0.01
	K2 = 0.03 
	C1 = K1*L*K1*L
	C2 = K2*L*K2*L

	(n,m) = imA.shape
	ua = 0
	ub = 0
	siga = 0
	sigb = 0 
	sigab = 0 
	#calculate mean of both images 
	for i in range(0,n):
		for j in range(0,m):
			ua = ua + imA[i][j]
			ub = ub + imB[i][j]
	ua = ua/(n*m)
	ub = ub/(n*m)
	
	#calculate sigma values 
	for i in range(0,n):
		for j in range(0,m):
			ai = imA[i][j]
			bi = imB[i][j]
			siga = siga + (ai-ua)*(ai-ua)
			sigb = sigb + (bi-ub)*(bi-ub)
			sigab = sigab + (ai-ua)*(bi-ub)
	
	siga = siga/(n*m)
	sigb = sigb/(n*m)
	sigab = sigab/(n*m)

	#use computed values to calculate ssim 
	ssim = (2*ua*ub + C1)*(2*sigab + C2)/((ua*ua - ub*ub + C1)*(ua*ua + ub*ub + C2))
	return ssim 
	
			
if __name__ == '__main__':
	imgA = cv2.imread('test_out20.jpg')
	imgB = cv2.imread('20_canny.jpg')
	imgA = cv2.cvtColor(imgA,cv2.COLOR_BGR2GRAY)
	imgB = cv2.cvtColor(imgB,cv2.COLOR_BGR2GRAY)
	cv2.imshow('img', imgA)                                   # Display
	cv2.waitKey(0)
	cv2.imshow('img', imgB)                                   # Display
	cv2.waitKey(0)
	mse = mse(imgA, imgB)
	print(mse)
	dice = dice(imgA, imgB)
	print(dice)
	ssim = ssim(imgA, imgB)
	print(ssim)
	

	
			
	
	
