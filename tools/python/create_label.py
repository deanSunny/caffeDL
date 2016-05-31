"""
Created on Wed Apr 06 09:29:23 2016

@author: sunny
"""

import os

def create_label(input_path, output_path, stype):
    label_file = 'optf_lab.txt'
    slabel_file = os.path.join(output_path, label_file)
    label_classes = 'optf_' + stype + '.txt'
    slabel_classes = os.path.join(output_path, label_classes)    
    
    if os.path.isdir(input_path):
        flist = os.listdir(input_path)
        for fl in flist:
            if fl == stype:
                detail = os.path.join(input_path, fl)
                subflist = os.listdir(detail)
                with open(slabel_file, 'w') as f, open(slabel_classes, 'w') as fc:
                    ftag = 1
                    for subf in subflist:
                        detail_classes = os.path.join(detail, subf)
                        if os.path.isdir(detail_classes):
                           f.write(subf + ' ' + str(ftag))
                           if ftag in range(101):
                               f.write('\n')
                
                           classes = os.listdir(detail_classes)
                           for cls in classes:
                               optname = os.path.join(detail_classes, cls)
                               fc.write(optname + ' ' + str(ftag) + '\n')
                                                          
                           ftag += 1     
                        else:
                            print 'no such subdir.'
    else:
        print 'no such root dir.'



if __name__ == '__main__':
    in_ = '/home/u514/DTask/data/optf_ucf101/optflow_L10'
    out_ = '/home/u514/DTask/data/optf_ucf101/optflow_L10/label'
    create_label(in_, out_, 'train')
    create_label(in_, out_, 'test')
    
    
