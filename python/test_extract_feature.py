'''
@author: Dean
@CAMALAB

2016-4-4
'''
#IMPORT
import os
import get_feature_f7_function

input_path = '/home/u514/DTask/1'
output_path = '/home/u514/DTask/1'
layername = 'fc7'

get_feature_f7_function.extract_features_from_CNN(input_path, output_path, layername)
