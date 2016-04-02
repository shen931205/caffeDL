'''
@author: Dean

2016-3-30

'''
import numpy as np
import caffe

import scipy.stats

class KDLLossLayer(caffe.Layer):
	'''
	Compute the KLD Loss.
	'''

	def setup(self, bottom, top):
		# check input pair
		if len(bottom) != 2:
			raise Exception('Need two inputs to compute distance.')

	def reshape(self, bottom, top):
		# check input dimesions match
		if bottom[0].count != bottom[1].count:
			raise Exception('Inputs must have the same dimesion.')
		
		self.diff = np.zeors_like(bottom[0].data, dtype=np.float32)
		top[0].reshape(1)

	def forward(self, bottom, top):
		count = bottom[0].count
		num = bottom[0].num
		loss = 0
		target = bottom[1].data
		input_data = bottom[0].data

		yi = softmax(input_data)
		loss = sci.stats.entropy(tartget, yi)	
		
		top[0].data[0] = loss / num

	def backward(self, bottom, propagate_down top):
		for i_ in range(2):
			if not propagate_down[_i]:
				continue
			

			bottom[_i].diff[...] = 


	def softmax(vector):
		sum_e = 0
		flag = 0
		yvector = np.zeros_like(vector, dtype=np.float32)
		for v in vector:
			sum_e += np.exp(v)
		for v_ in vector:
			y = np.exp(v_) / sum_e
			yvector[flag] = y
			flag += 1
		return yvector
