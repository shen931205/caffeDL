'''
@author: Dean
@CAMALAB HDU

2016-4-1
'''

from PIL import Image
import random

def crop_image(im, im_size, dim):
	'''
	Crop a image use given dim.
	'''
	#xsize, ysize = im.size
	xsize = int(im_size)
	ysize = int(im_size)
	xstart = random.randint(0, xsize - int(dim))
	ystart = random.randint(0, ysize - int(dim))
	crop_dim = (xstart, ystart, xstart + int(dim), ystart + int(dim))
	im_crop = im.crop(crop_dim)

	return im_crop
