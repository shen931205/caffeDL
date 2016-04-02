'''
@author: Dean
@CAMALAB

2016-4-2
'''
#import
import numpy as np
import scipy
import sys
import os

CAFFE_ROOT = '/home/u514/caffe-i/caffe-master/caffe/python'
sys.path.insert(0, CAFFE_ROOT)
import caffe

def extract_features_from_CNN(image_path_root, save_path_root, layername):
    '''
    Extract features from CNN.
    Default layers: the last but one fullconnect layer(e.g., fc7 in vgg_M_2048).

    image_path_root: the image folder (no subfolder).
    save_path_root: the target path you want to save.
    layername: the layer you want to extract features (e.g., fc7).
    '''
    # Set path to save features.
    feature_dir = os.path.join(save_path_root, 'feature')
    featureStr_dir = os.path.join(save_path_root, 'featureStr')
    classify_dir = os.path.join(save_path_root, 'classify')
    classify_output = os.path.join(classify_dir, 'classify_output.txt')
    outputfile = os.path.join(feature_dir, 'output')

    if not os.path.exists(feature_dir):
        os.system('sudo mkdir -p ' + feature_dir)
    if not os.path.exists(featureStr_dir):
        os.system('sudo mkdir -p ' + featureStr_dir)
    if not os.path.exists(classify_dir):
        os.system('sudo mkdir -p ' + classify_dir)
    if not os.path.exists(classify_output):
        os.system('sudo touch ' + classify_output)

    # Set parameters.
    files = os.listdir(image_path_root)
    i = 0
    j = 0
    count = 0
    allfeatures = []
    filename = []

    batch_size = 50
    raw_image_size = 256
    crop_size = 224

    net, transformer = init_caffe_net(3, raw_image_size, crop_size, batch_size)
    fileHandle = open(classify_output, 'a')

    for f in files:
        if os.path.isfile(os.path.join(image_path_root, f)):
            if i == batch_size:
                i = 0
                j += 1
                out = net.forward()
                featureData = net.blobs[layername].data
                probData = net.blobs['prob'].data
                # Predict the top-1 class.
                predict = np.argmax(probData, axisd = 1).reshape(batch_size, -1)

                for k in range(0, batch_size):
                    classnum = predict[k][0]
                    if len(allfeatures) == 0:
                        allfeatures = featureData[k].reshape(1, -1)
                    else:
                        tempfeature = featureData[k].reshape(1, -1)
                        allfeatures = np.vstack((allfeatures, tempfeature))
                    #- tag -#
                    fileHandle.write(str(filename[k]) + ' ' + str(classnum) + '\n')
                print np.shape(allfeatures)
                print '--------------- Have extracted ', j * batch_size, ' images. ---------------'

            filename.append(os.path.splitext(f)[0])
            # Load images.
            net.blobs['data'].data[i, ...] = transformer.preprocess(
                    'data', caffe.io.load_image(
                        os.path.join(image_path_root, f)))
            i += 1
            count += 1
        if (count - 1) % 2000 == 0 and (count - 1) != 0:
            np.save(outputfile + str(count - 1), allfeatures)
            print '-----# Save as ', outputfile + str(count - 1), ' in ', feature_dir
            allfeatures = []

    # Deal with the rest of images not in batch.
    print '--------------- Have extracted the rest ', i, ' images. ---------------'
    # the same operation in batch.
    out = net.forward()
    featureData = net.blobs[layername].data
    probData = net.blobs['prob'].data
    predict = np.argmax(probData, axis = 1).reshape(batch_size, -1)

    for k in range(0, i):
        classnum = predict[k][0]
        if len(allfeatures) == 0:
            allfeatures = featureData[k].reshape(1, -1)
        else:
            tempfeature = featureData[k].reshape(1, -1)
            allfeatures = np.vstack((allfeatures, tempfeature))
        #-- tag --#
        fileHandle.write(str(filename[k]) + ' ' + str(classnum) + '\n')
    np.save(outputfile + str(count), allfeatures)

    fileHandle.close()
    print '--------------- Extract features Done. ---------------'
def init_caffe_net(gpu_id, raw_image_size, crop_size, batch_size):
    '''
     Initialize caffe configuration
    '''
    caffe.set_mode_gpu()
    caffe.set_device(int(gpu_id)) # {0, 1, 2, 3} to four GPUs you want to choose.
    # The train_val.prototxt file defination.
    model_def = '/home/u514/caffe-i/caffe-master/caffe/models/vgg/vgg_2048/deploy-bak.prototxt'
    # The pre-trained model.
    caffemodel = '/home/u514/caffe-i/caffe-master/caffe/models/vgg/vgg_2048/pretrain_ilsvrc2012_vgg_2048.caffemodel'
    # The mean file of the image set used to trained the model.
    mean_file = '/home/u514/caffe-i/caffe-master/caffe/models/vff/vgg_2048/vgg_mean.npy'
    net = caffe.Net(model_def, caffemodel, caffe.TEST)

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1)) # (H,W,C) to (C,H,W)
    transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))
    transformer.set_raw_scale('data', int(raw_image_size))
    transformer.set_channel_swap('data', (2, 1, 0)) # RGB to BGR

    # Set batch size (default: 50).
    net.blobs['data'].reshape(int(batch_size), 3, int(crop_size), int(crop_size))
    return net, transformer

if __name__ == '__main__':

    image_input_folder = ''
    save_folder = ''
    layername = 'fc7'
    extract_features_from_CNN(image_input_folder, save_folder, layername)


