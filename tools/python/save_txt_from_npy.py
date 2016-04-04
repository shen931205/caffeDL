'''
@author: Dean
@CAMALAB

2016-4-3
'''
#import
import os
import numpy as np
from np_to_txt import print_txt

def _init_(input_path, output_path):
    '''
     Initialize the parameters.
     input_path: the path of the *.npy.
     output_path: the path of the *.txt you want to save.
    '''
    input_path = os.path.expanduser(input_path)
    output_path = os.path.expanduser(output_path)

    translate(input_path, output_path)

    print 'Done.'

def translate(_in, _out):
    flist = os.listdir(_in)
    for f in flist:
        if os.path.isfile(os.path.join(_in, f)):
            filename = os.path.splitext(f)[0]
            output_name = filename + '.txt'
            print_txt(os.path.join(_in, f), os.path.join(_out, output_name))

if __name__ == '__main__':
	in_ = '/home/u514/DTask/1/feature'
	out = '/home/u514/DTask/1/featureStr'
	_init_(in_, out)
