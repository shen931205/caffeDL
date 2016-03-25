import numpy as np
import matplotlib.pyplot as plt
import scipy 
import sys 
import os
import lmdb
import gc
sys.path.append('/home/u514/caffe-i/caffe-master/caffe/python')
import caffe

#dir
image_dir = '/home/u514/Lee/CUHKPhotoQualityDatabase/TEST/Good/'
#feature_dir = '/home/u514/Lee/CUHK/TEST/feature/Bad/5/'
feature_dir = '/home/u514/Lee/CUHK/TEST/feature/Good/originImageSet/'
#featureStr_dir = '/home/u514/Lee/CUHK/TEST/featureStr/Bad/5/'
featureStr_dir = '/home/u514/Lee/CUHK/TEST/featureStr/Good/originImageSet/'
#classify_path = '/home/u514/Lee/CUHK/TEST/classify/Bad/5/'
classify_path = '/home/u514/Lee/CUHK/TEST/classify/Good/originImageSet/'
#classify_dir = '/home/u514/Lee/CUHK/TEST/classify/Bad/5/classify_output.txt'
classify_dir = '/home/u514/Lee/CUHK/TEST/classify/Good/originImageSet/classify_output.txt'
#outputfile = '/home/u514/Lee/CUHK/TEST/feature/Bad/5/output'
outputfile = '/home/u514/Lee/CUHK/TEST/feature/Good/originImageSet/output'

if not os.path.isdir(feature_dir):
    os.system('sudo mkdir -p ' + feature_dir)
if not os.path.isdir(featureStr_dir):
    os.system('sudo mkdir -p ' + featureStr_dir)
if not os.path.exists(classify_path):
    os.system('sudo mkdir -p ' +classify_path)

####Caffe configuration####

caffe.set_mode_gpu()
caffe.set_device(1)

model_def = '/home/u514/caffe-i/caffe-master/caffe/models/vgg/vgg_2048/deploy-bak.prototxt'
caffemodel = '/home/u514/caffe-i/caffe-master/caffe/models/vgg/vgg_2048/pretrain_ilsvrc2012_vgg_2048.caffemodel'
mean_file = '/home/u514/caffe-i/caffe-master/caffe/models/vgg/vgg_2048/vgg_mean.npy'

net = caffe.Net(model_def, caffemodel, caffe.TEST) 

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2, 1, 0))

#set batchsize = 50
net.blobs['data'].reshape(50, 3, 224, 224)

####Get feature####

files = os.listdir(image_dir)
i = 0
j = 0
count = 0
allfeatures = []
fileNameNum = []
if not os.path.exists(classify_dir):
    os.system('sudo touch ' + classify_dir)
fileHandle = open(classify_dir, 'a')

featuretxt = feature_dir + 'feature_2048.txt'

#if not os.path.exists(featuretxt):
#    os.system('sudo touch ' + featuretxt)
#featureFile = open(featuretxt, 'a')

for f in files:
    if os.path.isfile(image_dir + f):
        if(i == 50):
	    i = 0
	    j = j + 1
	    
     	    out = net.forward()
	    fc7Data = net.blobs['fc7'].data
	    probData = net.blobs['prob'].data
            predict = np.argmax(probData, axis = 1).reshape(50, -1)

	    for k in range(0, 50):
		classnum = predict[k][0]
		if len(allfeatures) == 0:
                    allfeatures = fc7Data[k].reshape(1, -1)
		else:
                    feature = fc7Data[k].reshape(1, -1)
		    allfeatures = np.vstack((allfeatures, feature)) 
		#print allfeatures
		#print np.shape(feature)[0], np.shape(feature)[1]
		#featureFile.write(str(feature) + '\n')
			
		#tempData = feature.tobytes()
		#print fileNameNum[k][0]
		fileHandle.write(str(fileNameNum[k])+ '   ' + str(classnum) + '\n')
	    print np.shape(allfeatures)[0], np.shape(allfeatures)[1]
	    print '##################have extracted ',j*50,' images. ##################'
	fileNameNum.append(f.split('.')[0]) 
        net.blobs['data'].data[i,:,:,:] = transformer.preprocess('data', caffe.io.load_image(image_dir + f))  
        i = i + 1
        count = count + 1
    if ((count - 1) % 3000 == 0 and (count -1) != 0):
	np.save(outputfile + str(count - 1), allfeatures)
	print 'save as ', outputfile + str(count - 1)
	allfeatures = []

print ' ################## extracted the rest',i,' images. ##################'

out = net.forward()  
fc7Data = net.blobs['fc7'].data  
probData = net.blobs['prob'].data  
predict = np.argmax(probData, axis = 1).reshape(50,-1)

for k in range(0, i):
    classnum = predict[k][0]
    if len(allfeatures) == 0:
	allfeatures = fc7Data[k].reshape(1, -1)
    else:
	feature = fc7Data[k].reshape(1, -1)
	allfeatures = np.vstack((allfeatures, feature))
    #featureFile.write(str(feature) + '\n')
    fileHandle.write(str(fileNameNum[k][0])+ '   ' + str(classnum) + '\n')
    #np.save(outputfile, allfeatures)
#print allfeatures
np.save(outputfile + str(count), allfeatures)
#featureFile.close()  
fileHandle.close()

print 'Done.'



