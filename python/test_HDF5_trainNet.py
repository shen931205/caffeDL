'''
@author: Dean

2016-3-29

'''
#import 
import os
import sys
import numpy as np

CAFFE_HOME = '/home/u514/caffe-i/caffe-master/caffe/python'
sys.path.insert(0, CAFFE_HOME)
import caffe, h5py

from caffe import layers as L


def net(hdf5, batch_size):
	n = caffe.NetSpec()
	n.data, n.label = L.HDF5Data(batch_size = batch_size, source = hdf5, ntop = 2)
	n.ip1 = L.InnerProduct(n.data, num_output = 10, weight_filler=dict(type='xavier'))
