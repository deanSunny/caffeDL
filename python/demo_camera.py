#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# Rewritten by Dean
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
#import cv2.cv as cv
#Video part
import cfg_v

from video_wrapper import Video
import threading

CLASSES = ('__background__',
#           'aeroplane', 'bicycle', 'bird', 'boat',
#           'bottle', 'bus', 'car', 'cat', 'chair',
#           'cow', 'diningtable', 'dog', 'horse',
#           'motorbike', 'person', 'pottedplant',
#           'sheep', 'sofa', 'train', 'tvmonitor')
            'person_face')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}

def load_video(video_path):
    net = init_net()
    threads = []
#    capture = cv.CaptureFromFile(video_path)
#
#    nbFrames = int(cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_COUNT))
#
#    #CV_CAP_PROP_FRAME_WIDTH Width of the frames in the video stream
#    #CV_CAP_PROP_FRAME_HEIGHT Height of the frames in the video stream
#
#    fps = cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_FPS)

#    wait = int(1/fps * 1000/1)
#
#    duration = (nbFrames * fps) / 1000
#
#    print 'Num. Frames = ', nbFrames
#    print 'Frame Rate = ', fps, 'fps'
#    print 'Duration = ', duration, 'sec'
#
#    for f in xrange(nbFrames):
#        frameImg = cv.QueryFrame(capture)
#        print cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_POS_FRAMES)
##        cv.ShowImage("The Video", frameImg)
#        im = np.array(frameImg, dtype=np.uint64)
#        cv.WaitKey(wait)
#    cap = cv2.VideoCapture(video_path)
#    ret, frame = cap.read()
#    flag = 1
##    demo(net, frame)
#    while ret:
#        flag += 1
#        demo(net, frame, flag)
#        ret, frame = cap.read()
    
    cap = Video('c').camera_reader_item()
    if cap == 0:
        return
    flag = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
#        cv2.imshow('Camera', frame)
        th1 = threading.Thread(target=camera_show, args=(frame,))
#        th1.start()
        threads.append(th1)
        
        th2 = threading.Thread(target=demo, args=(net, frame, flag,))
#        th2.start()
        threads.append(th2)
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        
        key = cv2.waitKey(1)        
#        demo(net, frame, flag)
        if key & 0xFF == ord('q'):
            break
        flag += 1
    
    cap.release()

def camera_show(im):  
    try:
        cv2.imshow('Camera', im)
    except Exception, e:
        print 'Camera operator failed: {}.'.format(e)

def vis_detections(im, class_name, dets, flag,thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=12, color='white')

#    ax.set_title(('{} detections with '
#                  'p({} | box) >= {:.1f}').format(class_name, class_name,
#                                                  thresh),
#                  fontsize=14)

    plt.axis('off')
    plt.tight_layout()
    plt.draw()
#    plt.show()
#    plt.savefig('/home/dean/Files/result/test{}.jpg'.format(flag))

def demo(net, im, flag):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
#    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
#    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im, cls, dets, flag, thresh=CONF_THRESH)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')

    args = parser.parse_args()

    return args

def init_net():
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

#    prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
#                            'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    prototxt = os.path.join('/home/dean/Documents/py-faster-rcnn/models/WIDER_FACE/VGG16/faster_rcnn_end2end',
                            'test.prototxt')
#    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
#                              NETS[args.demo_net][1])
    caffemodel = os.path.join('/home/dean/Documents/py-faster-rcnn/output/faster_rcnn_end2end/voc_2007_train',
                              'vgg16_faster_rcnn_iter_50000.caffemodel')
    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)
    return net

if __name__ == '__main__':
#    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
#
#    args = parse_args()
#
##    prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
##                            'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
#    prototxt = os.path.join('/home/dean/Documents/py-faster-rcnn/models/WIDER_FACE/VGG16/faster_rcnn_end2end',
#                            'test.prototxt')
##    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
##                              NETS[args.demo_net][1])
#    caffemodel = os.path.join('/home/dean/Documents/py-faster-rcnn/output/faster_rcnn_end2end/voc_2007_train',
#                              'vgg16_faster_rcnn_iter_10000.caffemodel')
#    if not os.path.isfile(caffemodel):
#        raise IOError(('{:s} not found.\nDid you run ./data/script/'
#                       'fetch_faster_rcnn_models.sh?').format(caffemodel))
#
#    if args.cpu_mode:
#        caffe.set_mode_cpu()
#    else:
#        caffe.set_mode_gpu()
#        caffe.set_device(args.gpu_id)
#        cfg.GPU_ID = args.gpu_id
#    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
#
#    print '\n\nLoaded network {:s}'.format(caffemodel)
#
    # Warmup on a dummy image
#    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
#    for i in xrange(2):
#        _, _= im_detect(net, im)

#    im_names = ['000456.jpg', '000542.jpg', '001150.jpg',
#                '001763.jpg', '004545.jpg']
#    im_names = ['/media/dean/00011804000963B2/data/face_detection/WIDER_FACE/WIDER_test/images/10--People_Marching/10_People_Marching_People_Marching_2_21.jpg']
#    for im_name in im_names:
#        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
#        print 'Demo for data/demo/{}'.format(im_name)
#        demo(net, im_name)
    video_path = '/home/dean/Documents/pyPor/video/test2.avi'
    load_video(video_path)
#    plt.show()
