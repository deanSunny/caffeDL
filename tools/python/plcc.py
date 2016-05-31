# -*- coding: utf-8 -*-
"""
Created on Fri May 13 20:48:32 2016

@author: sunny
@CAMALAB
"""
import numpy as np

def plcc(data, label):
    num = len(label)
    cls_num = len(label[0])
    data_k = np.zeros(num)
    label_k = np.zeros(num)
    plcc_u = 0
    plcc_p_l = 0  
    plcc_p_d = 0
    
    for k in xrange(num):
        for cls in xrange(cls_num):
            data_k[k] += data[k][cls] * (cls + 1)
            label_k[k] += label[k][cls] * (cls + 1)
    mean_data = np.mean(data_k)
    mean_label = np.mean(label_k)
    
    plcc_u = np.sum((label_k - mean_label) * (data_k - mean_data))
    plcc_p_l = np.sum((label_k - mean_label) ** 2)
    plcc_p_d = np.sum((data_k - mean_data) ** 2)
    plcc = plcc_u * 1.0 / np.sqrt(plcc_p_l * plcc_p_d) 
    
    return plcc
    

if __name__ == '__main__':
    data = np.asarray([(0.5, 0.1, 0.01, 0.39), (0.4, 0.5, 0.001, 0.099)])
    label = np.asarray([(0.2, 0.5, 0.2, 0.1), (0.5, 0.1, 0.2, 0.2)])
    plcc = plcc(data, label)
    
    print plcc
    