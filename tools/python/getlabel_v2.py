'''
@author: Dean

2016-3-25

'''
import os
import sys
import argparse

from getlabel import get_classes_ind

##
def main(argv):
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'input_path',
		help = 'the path of input file folders.'
	)

	args = parser.parse_args()

	return args.input_path

##Get multilabel for dataset(such as UCF-101).

def get_multilabel(input_path):
	input_path = os.path.expanduser(input_path)
	video_classes = get_classes_ind(input_path) 
	for i in video_classes:
		if i != '__background__':
			sub_video_classes_path = os.path.join(input_path, i)
			sub_video_classes = get_classes_ind(sub_video_classes_path)
			print '--------------------{}---------------------- '.format(i)
			for j in sub_video_classes:
				if j != '__background__':
					optf_path = os.path.join(sub_video_classes_path, j)
					optf_source = os.listdir(optf_path)
					optf_ = sorted(optf_source)
					with open('multilabel.txt', 'a') as fid:
						for k in optf_:
							fid.write(os.path.join(optf_path, k)+ ' ' + str(video_classes[i]) + ' ' + str(sub_video_classes[j]) + '\n')
if __name__ == '__main__':
	_input = main(sys.argv)
	get_multilabel(_input)
