import math
import Image 
import Levenshtein 

''' Compares the Canny generated edge map to the training set edge map image using the following parameters imgt is the training image, imgc is the canny edge detection generated image'''

class ImCompare(object):

_pixel = 255
_colour = False

def __init__(self, imgt, imgc)
'''Saves a copy of the image objects'''

sizet, sizec = imgt.size, imgc.size

#rescale to a common size if not already the case 
if(sizet not= sizec):
	imgc = imgc.resize((sizet[0],sizet[1]), Image.BICUBIC)
	
#store image size 
self.x, self.y = sizet[0],sizet[1]

def _img_int(self, img):
'''Convert an image to a list of pixels '''
	x, y = img.size 
	for i in xrange(x):
		for j in xrange(y):
			yield img.hetpixel((i,j))

@property
def imgt_int(self):
''' returns tuple representing training image '''

	if not hasattr(self, '_imgt_int'):
		self._imgt_int = tuple(self._img_int(self._imgt))

	return self._imgt_int

@property 
def imgc_int(self):
''' returns tuple representing canny edge detection image ''' 
 	if not hasattr(self, '_imgc_int'):
		self._imgt_int = tuple(self._img_int(self._imgc))

	return self._imgc_int

@property 
def mse(self):
''' returns the mean square error between the two images.'''
	if not hasattr(self, '_mse'):
		temp = sum((a-b)**2 for a,b in zip(self.imgt_int, self.imgc_int))
		self._mse = float(temp)/self.x/self.y
	return self._mse

@property 
def psnr(self):
''' calculates the peak signal-to-noise ratio. '''
	if not hasattr(self,'_psnr'):
		self._psnr = 20*math.log(self._pixel/math.sqrt(self.mse), 10)
	return self._psnr





ct = 0
cf = 0
[m n] = size()

for i=1:m
	for j=1:n
		if ti(i,j)~=0 && ci(i,j)~=0
			ct=ct+1
		end
		if ti(i,j)~=0 && ci(i,j)==0 || ti(i,j)==0 && ci(i,j)~=0)
			cf=cf+1
		end
	end
end
