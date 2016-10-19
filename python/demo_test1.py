# -*- coding: utf-8 -*-
"""
Created on Sun Oct  5 17:12:11 2016

@author: dean
"""

import cfg_demo
import os
import sys
import cv2
import numpy as np
import argparse
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
#import demo_rec
import cPickle
from PIL import Image
from timer import Timer
from video_wrapper import Video
import caffe

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')
class_dict = dict(zip(range(21), CLASSES))
#print class_dict
def init_net_faster_rcnn(prototxt, caffemodel, gpu_id):
    '''
    '''
    cfg.TEST.HAS_RPN = True
    if not os.path.isfile(caffemodel):
        raise IOError('no such model exists, please check it out.')
    caffe.set_mode_gpu()
    caffe.set_device(int(gpu_id))
    cfg.GPU_ID = int(gpu_id)
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
#    net = init_net()
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)
    return net

def init_net_cnn(prototxt, caffemodel, gpu_id):
    '''
    '''
    caffe.set_mode_gpu()
    caffe.set_device(int(gpu_id))
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    net.blobs['data'].reshape(1, 3, 224, 224)
    
    return net
    
def parse_args():
    '''
    '''
    parser = argparse.ArgumentParser(description='Video object matching demo')

    parser.add_argument('--gpu', dest = 'gpu_id', type=int, default=0,
                        help='The used gpu id.')    
    parser.add_argument('--objtype', dest = 'type_id', type=int, default=0,
                        help='Chooes object type you want to detect, range of (1, 20) to {}'.format([tt for tt in CLASSES]))    
    parser.add_argument('--maxframe', dest = 'max_frame', default=999, type=int,
                        help='The max number of frames.')
    parser.add_argument('input', dest = 'img',
                        help='The input image for requirement.')
    parser.add_argument('video_input', dest = 'v_input',
                        help='The input video.')
    args = parser.parse_args()
    return args
#def get_require_obj(im):
#    args = parse_args()  
#    obj_type = args.type_id
#    if obj_type < 1 or obj_type > 20:
#        print 'Type ID out of range, please input in range from 1 to 20!'
#        return
#    else:
#        print 'Your choice is: {}'.format(class_dict[obj_type])
#    img_file = args.img
#    im = cv2.imread(img_file)
#
#       
#    
#    net = init_net()
#    timer = Timer()
#    timer.tic()
#    scores, boxes = im_detect(net, im)
#    timer.toc()
#    

def get_args():
    '''
    '''
    args = parse_args()
    return args
    

def get_required_obj(net, args):
    '''
    '''
    img_file = args.img
    im = cv2.imread(img_file)
    
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])
#    print scores, boxes
    return scores, boxes
    
def calc_required_obj_features(net, args):
    '''
    
    '''

    return features, cls 

def create_video_obj(net, args, v_path, cls_ind, max_frame):
    '''
    using faster-rcnn get object with per image.
    '''
    os.mkdir('cache')
    cache_file = 'cache/frame.pkl'
    cache_crop_file = 'cache/crop_frame.pkl'
    cache_dets_file = 'cache/dets.pkl'
    vw_cap = Video('v', v_path).cap_V
    max_frame_num = vw_cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
    if max_frame <= max_frame_num:
        max_frame_num = max_frame
    
    frame_list = []    
    ret, frame = vw_cap.read()
    tag = 1
    cache_crops = []
    cache_dets = []
    with open(cache_file, 'wb') as cf, open(cache_crop_file, 'wb') as ccf,        open(cache_dets_file, 'wb') as cdf:
        while ret is True:
            if tag > max_frame_num:
                break
            else:
                print 'The {} images are under processing.'.format(tag)
                tag += 1
            f_im = np.array(frame)
            frame_list.append(f_im)
            im = f_im[:, :, (2, 1, 0)]
            scores, boxes = im_detect(net, im)
            im_PIL = Image.fromarray(f_im)
            
            crop_im, dets = calc_dets(scores, boxes, im_PIL, cls_ind)
            cache_crops.append(crop_im)
            cache_dets.append(dets)
#                cPicle.dump(f_im, cc) #tag
            ret, frame = vw_cap.read()
            f_im = np.array(frame)
        vw_cap.release()
        frame_dict = dict(zip(range(1, max_frame_num + 1), frame_list))
#        dets_dict = dict(zip(range(1, max_frame_num + 1), ))
        cPickle.dump(frame_dict, cf)
        cPickle.dump(cache_crops, ccf)
        cPickle.dump(cache_dets, cdf)
        
    return cache_crop_file, cache_file, cache_dets

def calc_dets(scores, boxes, im, cls_ind, CONF_THRESH=0.8):
    '''
    calculate bouding boxes within per image.
    '''
    NMS_THRESH = 0.3
#    for cls_ind, cls in enumerate(CLASSES[1:]):
#        print cls_ind, cls
#        cls_ind += 1
    cls_boxes = boxes[:, 4 * cls_ind: 4 * (cls_ind + 1)]
    cls_scores = scores[:, cls_ind]
    dets = np.hstack((cls_boxes,
                      cls_scores[:, np.newaxis])).astype(np.float32)
    keep = nms(dets, NMS_THRESH)
    dets = dets[keep, :]  
    
    inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
    crop_im = []
    if len(inds) == 0:
        return crop_im
#        im = im[:, :, (2, 1, 0)]
    
    
    for i in inds:
        origin_im = np.zeros((224, 224, 3), dtype=np.uint8)
        region = dets[i, :4]
        print region
        tmp_im = im.crop(region)
#            tmp_im = np.array(tmp_im)
        w, h = tmp_im.size
#            if w / 224 >= h / 224:
        if w >= h:
            rate = crop_resize_rate(w, 224)
        else:
            rate = crop_resize_rate(h, 224)
        resize_region = (w * rate, h * rate)   
        resize_im = tmp_im.resize(resize_region, Image.ANTIALIAS)
        resize_im = np.array(resize_im) + origin_im
        crop_im.append(resize_im)
    
    return crop_im, dets

def get_data_from_cache(cache_file):
    '''
    get data from binary files.
    '''
    if not os.path.exists(cache_file):
        raise IOError('no such file exists, please check it out.')
    
    return cPickle.load(cache_file)
           
def crop_resize_rate(len_, sta_input):
    '''
    return the rate of image which should be resized.
    '''
    rate = float(sta_input) / float(len_)
    return rate

def calc_video_obj_features(net, im_file, crop_file):
    '''
    '''
    features = []
    crop_im = get_data_from_cache(crop_file)
    feature = extract_feature(net, 'fc7')
    return features
    
def calc_distance(target_features, proposal_features):
    '''
    '''
    
    return dist

def extract_feature(net, layer, im):
    '''
    extract features from CNN.
    '''
    net.blobs['data'].data = im
    net.forward()
    feature = net.blobs[layer].data
    return feature

def evaluation(dist):
    '''
    '''
    pass 
        
def user():
    '''
    '''
    pass
    
def demo():
    '''
    demo for search.
    '''
    args = get_args()
    #get required OBJ feature
    prototxt_cnn = ''
    caffemodel_cnn = ''        
    net_cnn = init_net_cnn(prototxt_cnn, caffemodel_cnn, args.gpu_id)
    
    re_obj_f, re_cls = calc_required_obj_features(net_cnn, args)    
    
    prototxt_faster_rcnn = ''
    caffemodel_faster_rcnn = ''
    net_faster_rcnn = init_net_faster_rcnn(prototxt_faster_rcnn, 
                                           caffemodel_faster_rcnn, args.gpu_id)
    crop_im = create_video_obj(net_faster_rcnn, args, args.v_input, re_cls, args.max_frame)
    
    net_cnn = init_net_cnn(prototxt_cnn, caffemodel_cnn, args.gpu_id)
    proposal_features = calc_video_obj_features(net_cnn, crop_im)
    
    #calculate distance.    
    dist = calc_distance(re_obj_f, proposal_features)
    
    
    
#    get_required_obj(net, args)
#    get_video_obj(net, args)
      
if __name__ == '__main__':
    demo()
#    calc_dets("", "")
    
    


