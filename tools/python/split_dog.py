# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 19:18:40 2016

@author: sunny
@CAMALAB
"""

import os
import linecache
from endWith import endWith


def split_dog(path):
    path = os.path.expanduser(path)
    if not os.path.isdir(path):
        print 'No such dir.'
        return 
    
    list_dir = os.listdir(path)
    ext_ = endWith('.txt')
    f_list = filter(ext_, list_dir)
    for f_name in f_list:
        print f_name, 'is under splitting.'
        f_name_pre = os.path.splitext(f_name)[0]
        if not os.path.exists(os.path.join(path, 'split', f_name_pre)):
            os.makedirs(os.path.join(path, 'split', f_name_pre))
        
        save_path = os.path.join(path, 'split', f_name_pre)        
        split_num = 1      
        num = 1
        split_default = 2000000         #each split file includes lines.
        s_file = os.path.join(path, f_name)
        if os.path.isfile(s_file):
            with open(s_file, 'rb') as f:
                count = 0
                while True:
                    buffer = f.read(8192*1024)
                    if not buffer:
                        break
                    count += buffer.count('\n')
            f.close()
        while(count >= num + split_default * (split_num - 1)):
            with open(os.path.join(save_path, 'split' + str(split_num) + '.txt'), 'a') as sf:
                
                inline = linecache.getline(s_file, num + split_default * (split_num - 1))
                inline_ = inline.split('\t')
                newline = inline_[0] + ' ' + inline_[1] + ' ' + inline_[2] + ' ' + inline_[3]
                sf.write(newline)
                num += 1
                if num > split_default:
                    num = 1
                    split_num += 1
                    print 'split {} done.'.format(split_num - 1)
            sf.close()
                
        #print linecache.getline(s_file, 2)    
                
        
if __name__ == '__main__':
    path = 'E:\\Data\\CFC'
    split_dog(path)
    print '--- Done. ---'

