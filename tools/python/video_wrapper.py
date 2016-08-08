# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 20:40:09 2016

@author: dean
"""
#import
import numpy as np
import cv2
import os


class Video(object):
    '''
    A simple video r/w wrapper for guiding.    
    '''
    def __init__(self, type_, path=None):

        type_def = ['v', 'c']
        type_dict = dict(zip(type_def, xrange(len(type_def))))
        if type_ not in type_def:
            print 'Unsigned type, please reinput.(v for video, c for camera)'
            return
        if type_dict[type_] == 0:
            assert os.path.exists(path), \
                'No such file: {}'.format(path) 
            self.path = path
            self.cap_V = cv2.VideoCapture(self.path)
        elif type_dict[type_] == 1:
            self.cap_C = cv2.VideoCapture(0)
        
    def video_reader(self):
        
        if not self.cap_V.isOpened():
            print 'Video: {} open failure.'.format(self.path)
            return
        else:
            print 'Video: {} open success.'.format(self.path)
        ret, frame = self.cap_V.read()
        while ret:
            ret, frame = self.cap_V.read()
        self.cap_V.release()
        print 'Video: {} close.'.format(self.path)
    
    def video_fps(self):
        fps = self.cap_V.get(cv2.cv.CV_CAP_PROP_FPS)
        print fps
    
    def camera_reader(self):

        if not self.cap_C.isOpened():
            print 'Camera open failure.'
            return 
#        cv2.namedWindow('Camera')
        
        flag = 1
        self.camera_writer()
        while True:            
            ret, frame = self.cap_C.read()
            if not ret:
                break
            cv2.imshow('Camera', frame)
            key = cv2.waitKey(1)
            if key == ord('s'):
#                cv2.imwrite('camera/sc_{}.jpg'.format(flag), frame)
                pass
            elif key & 0xFF == ord('q'):
                break
            flag += 1
#            self.videoWriter.write(frame)
            ff = cv2.flip(frame, 0)
            self.out.write(ff)
        
        self.cap_C.release()
        self.out.release()
        cv2.destroyAllWindows()
        
    def camera_reader_item(self):
        if not self.cap_C.isOpened():
            print 'Camera open failure.'
            return 0
        else:
            return self.cap_C
    
    def camera_writer(self):
#        fps = self.cap_C.get(cv2.cv.CV_CAP_PROP_FPS)
        fps = 30.0
        size = (1280, 720)
#        self.videoWriter = cv2.VideoWriter('/home/dean/Files/result/test_canera.mp4',
#                              cv2.cv.CV_FOURCC('M','J','P','G'),
#                                fps, size)
#        print size
#        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fourcc = cv2.cv.CV_FOURCC(*'XVID')
        self.out = cv2.VideoWriter('/home/dean/Data/result/test_video2.avi',
                                   fourcc, fps, size)
if __name__ == '__main__':
    v1 = Video('c')
    v1.camera_reader()
#    v1.camera_writer()

    
#    video_path = '/home/dean/Documents/pyPro/video/test.avi'
#    v2 = Video('v', video_path)
#    v2.video_fps()
#    v2 = Video('v', path=None)
#    v2.video_reader()
    