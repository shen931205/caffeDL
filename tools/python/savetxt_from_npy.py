import os
import numpy as np
import sys
from np_to_txt import print_txt
##dir
text_root_path = '/home/u514/Zhang/AVA/featureStr/originImageSet/'
npy_root_path = '/home/u514/Zhang/AVA/feature/originImageSet/'

##get *.npy

title = xrange(2, 194, 2)
'''
npy_test = npy_root_path + 'output2000.npy'

test1 = np.load(npy_test)
print test1
im_num = np.shape(test1)[0]
classes = np.shape(test1)[1]
os.system('sudo touch ' + text_root_path + 'test1.txt')
fileHandle = open(text_root_path + 'test1.txt', 'a')
for i in range(im_num):
    for j in range(classes):
	temp = test1[i][j]
	fileHandle.write(str(temp) + ' ')
    fileHandle.write('\n')

## print to *.txt
def print_txt(inputfile, outputfile):
    print '###############----Loading file: ', inputfile, '----###############' 
    scores = np.load(inputfile)
    im_num = np.shape(scores)[0]
    classes = np.shape(scores)[1]

    os.system('sudo touch ' + outputfile)
    fileHandle = open(outputfile, 'a')

    for i in range(im_num):
        for j in range(classes):
            temp = test1[i][j]
            fileHandle.write(str(temp) + ' ')
        fileHandle.write('\n')
    fileHandle.close()
    print ('###############----Saved in: ', outputfile, '----###############')
    
'''

## prepare for inputs

for i in title:
    npy_name = 'output' + str(i) + '000.npy'
    txt_name = 'output' + str(i) + '000.txt'
    input_file = npy_root_path + npy_name
    output_file = text_root_path + txt_name
     
    print_txt(input_file, output_file)

input_last = npy_root_path + 'output193076.npy'
output_last = text_root_path + 'output_last.txt'

print_txt(input_last, output_last)

print '###############----Done.----###############'


