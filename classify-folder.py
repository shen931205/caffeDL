import numpy as np
import matplotlib.pyplot as plt
import scipy 
import sys 
import os
import lmdb
import gc
import argparse
import time
import cv2

import caffe



####dir####

#outputfile = '/home/u514/Zhang/test484829/feature/output'
outputfile = '/home/u514/caffe-i/caffe-master/caffe/python/testdemo/testfolder1.txt'
root_path = '/home/u514/caffe-i/caffe-master/caffe/python/testdemo/lib/ucf101/'

if root_path not in sys.path:
    sys.path.insert(0, root_path)
    
from ucf101_init import getlabels

####get input####

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_folder",
        help = "input the Image folder you want to test."
    )
    parser.add_argument(
        "batch_size",
        help = "set the batch size suitably."
    )
    args = parser.parse_args()

    return args.input_folder, args.batch_size

####draw the result####

def draw_result(classind, im, tag):
    label_path = '/home/u514/caffe-i/caffe-master/caffe/python/testdemo/lib/ucf101/labels.txt'
    out_path = '/home/u514/caffe-i/caffe-master/caffe/python/testdemo/drawtest1/'
    
    if not os.path.exists(out_path):
        os.system('sudo mkdir ' + out_path)
    
    labels = getlabels(label_path)
    #print labels[int(classind)]

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    '''
    fig = plt.figure(tag, figsize=(12, 12))
    ax = fig.add_subplot(111)
    '''
    ax.imshow(im, aspect='equal')
    ax.set_title(('classification result: {}').format(labels[int(classind)]), fontsize=14)
    
    plt.axis('off')    
    plt.draw()
    plt.tight_layout()
    plt.savefig(out_path + str(tag) + '.jpg', dpi=50)
    #plt.close(tag)

####classification####

def classify(input_folder, bs):
    
    batch_size = int(bs)
    if batch_size > 200:
        print 'Batch size is too big to set as {:s}, please reset!'.format(batch_size)
        sys.exit()
    
    ####Caffe configuration####
    caffe_start = time.time()
    caffe.set_mode_gpu()

    model_def = '/home/u514/caffe-i/caffe-master/caffe/models/vgg/vgg_2048/ucf101-deploy.prototxt'
    caffemodel = '/home/u514/caffe-i/caffe-master/caffe/models/vgg/vgg_2048/scnn_finetune_vgg_2048_0.5_iter_177730.caffemodel'
    mean_file = '/home/u514/caffe-i/caffe-master/caffe/models/vgg/vgg_2048/ucf101_mean.npy'

    net = caffe.Net(model_def, caffemodel, caffe.TEST) 

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))
    transformer.set_raw_scale('data', 255)
    transformer.set_channel_swap('data', (2, 1, 0))

    #set batchsize

    net.blobs['data'].reshape(batch_size, 3, 224, 224)

    print 'Use model_deploy : {:s}\n'.format(model_def), 'Use model : {:s}\n'.format(caffemodel)
    print 'Net initialization Done.Time : {:.2f}'.format(time.time() - caffe_start)

    ####Get feature####
    files = os.listdir(input_folder)
            
    i = 0
    j = 0
    count = 0
    fileName = []

    if not os.path.exists(outputfile):
        os.system('sudo touch ' + outputfile)


    for f in files:

        if os.path.isfile(input_folder + f):
            if(i == batch_size):
                i = 0
                j = j + 1
                
                out = net.forward()
                probData = net.blobs['prob'].data
                predict = np.argmax(probData, axis = 1).reshape(batch_size, -1)

                for k in range(0, batch_size):
                    classnum = predict[k][0]
                    tag = k + (j-1) * batch_size
                    print tag
		    im = cv2.imread(input_folder + fileName[tag])
                    draw_result(classnum, im, tag)

                    
                print '##################have extracted ',j*batch_size,' images. ##################'


            net.blobs['data'].data[i,:,:,:] = transformer.preprocess('data', caffe.io.load_image(input_folder + f))  
            i = i + 1
            fileName.append(f)    
            count = count + 1

    print '################## extracted the rest',i,' images. ##################'

    out = net.forward()   
    probData = net.blobs['prob'].data  
    predict = np.argmax(probData, axis = 1).reshape(batch_size,-1)

    for k in range(0, i):
        classnum = predict[k][0]
        tag = k + j * batch_size 
        print tag
        im = cv2.imread(input_folder + fileName[tag])
        draw_result(classnum, im, tag)
    
    print 'Done.'

if __name__ == '__main__':
    t = time.time()
    folder, batch_size = main(sys.argv)
    classify(folder, batch_size)

    print 'Mission complete!---------------Time: {:.2f}.'.format(time.time() - t)

