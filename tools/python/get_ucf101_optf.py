# -*- coding: utf-8 -*-
"""
Created on Wed Apr 06 11:19:39 2016
@author: sunny
@CAMALAB
"""

import h5py
import os
import numpy as np
from PIL import Image
from random import shuffle

from imagecroper import crop_image

def transform_image(im, im_size, crop_dim):
    im = im.resize((int(im_size), int(im_size)))
    im = crop_image(im, int(im_size), int(crop_dim))
    im = np.asanyarray(im, dtype=np.float32)
    #im = im[:, :, ::-1]
    #im = im.transpose((2, 0, 1))
    return im

def split_label(obj):
	return int(obj.split('v')[0])

def create_optical_HDF5_db(image, label, save_path, stype):
    s_type = ['train', 'test']
    im_list_num = [9537, 3783]
    sstype = dict(zip(s_type, im_list_num))
    #im_source_root = os.path.join(image, stype)  
    im_size = 256
    crop_dim = 224
    f = h5py.File(os.path.join(save_path, stype, [stype + '.h5'][0]), 'w')
    f.create_dataset('data', (sstype[stype], 20, crop_dim, crop_dim), dtype=np.float32)
    f.create_dataset('label', (sstype[stype], 1), dtype='i8')
    
    if stype not in s_type:
        print 'not train or test, please retry.'
        return
    
    im_flag = 0
    for im_ in image:
        im_list = os.listdir(im_)
	im_list = sorted(im_list, key=split_label)
        channel_flag = 0
        lb_ = np.asarray(label, dtype=np.uint8)
        for im_d in im_list:
            im_source = os.path.join(im_, im_d)
            im = Image.open(im_source)
            im = transform_image(im, im_size, crop_dim)
            f['data'][im_flag, channel_flag, ...] = im
            
            channel_flag += 1
        f['label'][im_flag] = lb_[im_flag]
	if im_flag % 20 == 0 and im_flag != 0:
		print '--------{} optical flow done.--------'.format(im_flag/20)
	im_flag += 1
    
    f.close()
#    shuffle(label)
#    im_flag = 0
#    
#    for lab in label:
#        filename = lab.split(' ')[0]
#        target = lab.split(' ')[1]
#        im_source = os.path.join(im_source_root, filename)
#        if os.path.isfile(im_source):
#            im = Image.open(im_source)
#            im = transform_image(im, im_size, crop_dim)
#            f['data'][im_flag] = im
#            f['label'][im_flag] = target
#            im_flag += 1
            
def get_optf_label(path):
    im = []
    label = []    
    im_label = []
    with open(path, 'r') as f:
        #shuffle(f)
        for fline in f:
            im_label.append(fline)
    shuffle(im_label)
    for im_ in im_label:
        im_ = im_.strip()
        im.append(im_.split(' ')[0])
        label.append(im_.split(' ')[1])
    return im, label
    
if __name__ == '__main__':
	path = '/home/u514/DTask/data/optf_ucf101/optflow_L10/label/optf_test.txt'
	save_path = '/home/u514/DTask/data/optf_ucf101/optflow_L10/h5data'
	im, label = get_optf_label(path)
	create_optical_HDF5_db(im, label, save_path, 'test')
