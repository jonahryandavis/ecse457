import math
import Image 
import cv2
import sys 
import numpy as np
from scipy import signal
from scipy import ndimage
import pylab


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

def fspecial_gauss(size, sigma):
    	"""Function to mimic the 'fspecial' gaussian MATLAB function
	from: https://github.com/mubeta06/python/blob/master/signal_processing/sp/gauss.py
    	"""
    	x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    	g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
	return g/g.sum()


def ssim(imA, imB):
	_pixel = 255
	#computes structural similarity index between two images
	imA = imA.astype(np.float64)
	imB = imB.astype(np.float64)	
	L = 255
	K1 = 0.01
	K2 = 0.03 
	C1 = K1*L*K1*L
	C2 = K2*L*K2*L
	size = 11
	sigma = 1.5
	window = fspecial_gauss(size,sigma)

	
	ua = signal.fftconvolve(window, imA, mode='valid')
	ub = signal.fftconvolve(window, imB, mode='valid')
	siga_sq = signal.fftconvolve(window, imA*imA, mode='valid') - ua*ua
	sigb_sq = signal.fftconvolve(window, imB*imB, mode='valid') - ub*ub
	sigab = signal.fftconvolve(window, imA*imB, mode='valid') - ua*ub 



	#use computed values to calculate ssim 
	ssim_map = (2*ua*ub + C1)*(2*sigab + C2)/((ua*ua + ub*ub + C1)*(siga_sq + sigb_sq + C2))
	
	return ssim_map.mean()

			
if __name__ == '__main__':
	#imgA = cv2.imread('test_images/test_out20.jpg',0)
	#imgB = cv2.imread('test_images/20_canny.jpg',0)
	#mse = mse(imgA, imgB)
	#print(mse)
	#dice = dice(imgA, imgB)
	#print(dice)
	#ssim = ssim(imgA, imgB)
	#print(ssim)
	img = cv2.imread('test_images/20_masked.jpg',0)
	sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
	sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
	sobel = np.sqrt(np.square(sobelx)+np.square(sobely))

	# ## Training Set

	# In[14]:

	# Transforming the feature set in the appropriate form for sklearn
	X1=img.reshape(-1) #Feature 1
	X2=sobel.reshape(-1) # Feature 2
	train=range(0,255)
	print(X1)
	

	
			
	
	
