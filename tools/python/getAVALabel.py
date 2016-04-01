'''
@author: Dean
@CAMALAB HDU

2016-3-29
'''
import h5py
import os
import numpy as np
from PIL import Image
from cv2 import imread
import scipy.misc

from imagecroper import crop_image 

def load_mat(mat_path, mat_name):
	h5file = h5py.File(mat_path)
	h5array = h5file[mat_name]
	if mat_name == 'ScorePrb':
		nparray = np.asarray(h5array, dtype=np.float32)
	else:
		nparray = np.asarray(h5array, dtype=np.uint32)
	return nparray


def create_HDF5_db(image_root_path, image, label, save_path, stype):
	set_type = ['train', 'test', 'val']
	set_num = [96537, 57923, 38615]
	if stype not in set_type:
		print 'Not available setType you set.'
		return 
	s_type = dict(zip(set_type, set_num))
	im_size = 256
	crop_dim = 227
	f = h5py.File(os.path.join(save_path, stype, [stype + '.h5'][0]), 'w')
 	# 193076 data, each has (256, 256, 3)-dim
	f.create_dataset('data', (s_type[stype], 3, crop_dim, crop_dim), dtype=np.float32)
	# Data's labels, each is a 10-dim vector
	f.create_dataset('label', (s_type[stype], 10), dtype=np.float32)
	
	# load images from *.jpg in image_path
	_im_flag = 0
	
	if stype == 'train':
		data_range = (0, s_type[stype])
	elif stype == 'test':
		data_range = (set_num[0]+set_num[2], set_num[0]+set_num[1]+set_num[2])
	elif stype == 'val':
		data_range = (set_num[0], set_num[0]+set_num[2])

	for _im in xrange(data_range[0], data_range[1]):
		current_im_path = os.path.join(image_root_path, (str(image[_im][0]) + '.jpg'))
		if os.path.exists(current_im_path):
			im_source = Image.open(current_im_path)
			im = im_source.resize((im_size, im_size))	
			im = crop_image(im, im_size, crop_dim) # crop images
			im = np.asarray(im, dtype=np.float32)
			im = im[:, :, ::-1]		# to BGR
			im = im.transpose((2, 0, 1)) # (H,W,C) to (C,H,W) 
			f['data'][_im_flag] = im
			f['label'][_im_flag] = label[_im]
			_im_flag += 1
		if _im % 2000 == 0 and _im != 0:
			print '-----------------{} images done.'.format(_im)
	f.close()		
	

if __name__ == '__main__':
	root_path = '/home/u514/DTask/sunnyMaster/data/AVA'
	label_path = os.path.join(root_path, 'ScorePrb.mat')
	image_path = os.path.join(root_path, 'ImageID.mat')
	
	raw_image_path = '/home/u514/DTask/data/AVA/originImageSet'


	labels = load_mat(label_path, 'ScorePrb')
	labels = np.transpose(labels)
	raw_images_name = load_mat(image_path, 'ImageID')
	create_HDF5_db(raw_image_path, raw_images_name, labels, '/home/u514/DTask/data/AVA', 'train')
	create_HDF5_db(raw_image_path, raw_images_name, labels, '/home/u514/DTask/data/AVA', 'test')
	create_HDF5_db(raw_image_path, raw_images_name, labels, '/home/u514/DTask/data/AVA', 'val')
	
	print 'Done.'
		




