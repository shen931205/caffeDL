#!/usr/bin/env sh

ROOT=/home/u514
TOOLS=$ROOT/caffe-i/caffe-master/caffe/build/tools
DATA=$ROOT/DTask/rcnn/data
echo "please wait..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    $ROOT/DTask/ \
    $DATA/test.txt \
    $DATA/outputIMDB/bbox_test 

echo "Done."

