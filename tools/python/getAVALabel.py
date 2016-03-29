'''
@author: Dean

2016-3-29
'''
import h5py
import os
import numpy as np
from cv2 import imread
import scipy.misc

def load_mat(mat_path, mat_name):
	h5file = h5py.File(mat_path)
	h5array = h5file[mat_name]
	if mat_name == 'ScorePrb':
		nparray = np.asarray(h5array, dtype=np.float32)
	else:
		nparray = np.asarray(h5array, dtype=np.uint32)
	return nparray


def create_HDF5_db(image_root_path, image, label, save_path, len_data):
	f = h5py.File(os.path.join(save_path, 'train.h5'), 'w')
 	# 193076 data, each has (256, 256, 3)-dim
	f.create_dataset('data', (int(len_data),3, 256, 256), dtype=np.float32)
	# Data's labels, each is a 10-dim vector
	f.create_dataset('label', (int(len_data), 10), dtype=np.float32)
	
	# load images from *.jpg in image_path
	_im_flag = 0
	for _im in xrange(96538):
		current_im_path = os.path.join(image_root_path, (str(image[_im][0]) + '.jpg'))
		if os.path.exists(current_im_path):
			im = imread(current_im_path)
			im = np.asarray(im, dtype=np.float32)
			im = scipy.misc.imresize(im, (256,256))
			im = np.reshape(im, (3, 256, 256))
			f['data'][_im_flag] = im
			f['label'][_im_flag] = label[_im]
			_im_flag += 1
		if _im % 2000 == 0 and _im != 0:
			print '-----------------{} images done.'.format(_im)
	f.close()		
	

if __name__ == '__main__':
	root_path = '../../data/AVA'
	label_path = os.path.join(root_path, 'ScorePrb.mat')
	image_path = os.path.join(root_path, 'ImageID.mat')
	
	raw_image_path = '/home/u514/DTask/data/AVA/originImageSet'


	labels = load_mat(label_path, 'ScorePrb')
	labels = np.transpose(labels)
	raw_images_name = load_mat(image_path, 'ImageID')
	create_HDF5_db(raw_image_path, raw_images_name, labels, '/home/u514/DTask/data/AVA', 96538)

	print 'Done.'
		




