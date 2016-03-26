'''
@author: Dean

2016-3-25

'''
import sys
import os
import argparse
##Get the argument from user.

def main(argv):
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'input_path',
		help = 'the path of input file folders.'
	)
	parser.add_argument(
		'output_file',
		default = './output.txt',
		help = 'the path of output file to save as.'
	)

	args = parser.parse_args()

	return args.input_path, args.output_file

##Get the classes in input path.

def get_classes_ind(input_path):
	input_path = os.path.expanduser(input_path)
	classes_ind = []
	if os.path.exists(input_path):
		classes_ind = os.listdir(input_path)
		classes_ind = sorted(classes_ind)
		classes_ind.insert(0, '__background__')
		classes_ind = dict(zip(classes_ind, xrange(102)))
	#print 'get_class_ind done.'
	return classes_ind

if __name__ == '__main__':
	_input, _output = main(sys.argv)
	get_classes_ind(_input, _output)

	print 'Done.'
