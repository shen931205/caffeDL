import numpy as np
import os


def print_txt(inputfile, outputfile):
    print '###############----Loading file: ', inputfile, '----###############'
    scores = np.load(inputfile)
    im_num = np.shape(scores)[0]
    classes = np.shape(scores)[1]

    os.system('sudo touch ' + outputfile)
    fileHandle = open(outputfile, 'a')

    for i in range(im_num):
        for j in range(classes):
            temp = scores[i][j]
            fileHandle.write(str(temp) + ' ')
        fileHandle.write('\n')
    fileHandle.close()
    print '###############----Saved in: ', outputfile, '----###############'

