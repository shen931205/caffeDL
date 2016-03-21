import sys
import json
import time
import pickle
import scipy.misc
import skimage.io

sys.path.append('./caffe/python')

import caffe

import numpy as np
import os.path as osp

from xml.dom import minidom
from random import shuffle
from threading import Thread
from PIL import Image

from tools import SimpleTransformer


class TemproalDataLayerSync(caffe.Layer):
    '''
    This is a datalayer for temproal cnn in two-stream cnn used for training the optical flow on
    UCF-101 and Hmdb-51 dataset.
    '''
    def setup(self, bottom, top):
	
		self.top_names = ['data', 'label']
		# === Read input parameters ===
	
		params = eval(self.param_str)

		#Check the parameters for validity.
		check_params(params)
	
		self.batch_size = params['batch_size']
		self.channels = params['channels']	
		self.height = params['im_shape'][0]
		self.width = params['im_shape'][1]

		self.batch_loader = BatchLoader(params, None)

		top[0].reshape(
	    	self.batch_size, self.channels, self.height, sefl.width)
			# Note the channels (UCF-101 has 101 classes)
			top[1].reshape(self.batch_size, 101)

		print_info("TemproalDataLayerSync", params)
    
	def forward(self, bottom, top):
		'''
		Load data.
		'''
		for itt in range(self.batch_size):
			im, label = self.batch_loader.load_next_image()

			top[0].data[itt, ...] = im
			top[1].data[itt, ...] = label

    def backward(self, bottom, top):
		pass

    def reshape(self, bottom, top):
		pass



class BatchLoader(object):
    '''
    This class is used to load images.
    Images can either be loaded singly, or in a batch(This is the useful part in temproal cnn on optical flow.)
    '''
	def __init__(self, params, result):
		self.result = result 
		self.batch_size = params['batch_size']
		self.im_shape = params['im_shape']

		self._cur = 0
	def load_next_image(self):
		'''
		Load the next image in a batch.
		'''
		
		if self._cur == len(self.indexlist):
			self._cur =0
			suffle(self.indexlist)

		index = self.indexlist(self._cur)

		#Choose the image type {e.g., .png, .jpg, .JPEG}.
		im_file_name = index + '.JPEG'
		im = np.asarry(Image)


def check_params(params):
	'''
	A utility function to check the parameters for the data layers.
	'''
	assert 'split' in params.keys(
		), 'Params must include split(train, val or test).'

	required = ['batch_size', 'channels', 'im_shape']

	for r in required:
		assert r in params.keys(), 'Params must include {}'.format(r)


def print_info(name, params):
	'''
	Output some info regarding the class.
	'''
	print "{} initialized for split: {}, with bs: {}, im_shape: {}.".format(
		name,
		params['split'].
		params['batch_size'],
		params['im_shape'])










