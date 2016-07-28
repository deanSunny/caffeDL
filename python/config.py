'''
@author: Dean
@CAMALAB
@2016-7-22
'''
#import
import os, sys

CAFFE_PATH = '/home/u514/caffe-i/caffe-master/caffe/python'

def add_path(path):
	if path not in sys.path:
		sys.path.insert(0, path)

add_path(CAFFE_PATH)
