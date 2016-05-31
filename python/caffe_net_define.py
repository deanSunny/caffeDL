'''
@author: Dean
@CAMALAB

2016-4-5
'''
#import 
import sys

CAFFE_ROOT = '/home/u514/caffe-i/caffe-master/caffe/python'
sys.path.insert(0, CAFFE_ROOT)
from caffe import layers as L, params as P, to_proto

def conv_relu(bottom, ks, nout, stride = 2, pad = 0, group = 1):
	conv = L.Convolution(bottom, kernel_size = ks, stride = stride, 
			num_output = nout, pad = pad, group = group)
	return conv, L.ReLU(conv, in_place = True)

def fc_relu(bottom, nout):
	fc = L.InnerProduct(bottom, num_output = nout)
	return fc, L.ReLU(fc, in_place = True)

def max_pool(bottom, ks, stride = 1):
	return L.Pooling(bottom, pool = P.Pooling.MAX, kernel_size = ks, stride = stride)

def alexnet_ava_finetuning(hdf5, batch_size = 256, include_acc = False):
	data, label = L.HDF5Data(source = hdf5, batch_size = batch_size, ntop = 2) # no transform_param in HDF5.

	conv1, relu1 = conv_relu(data, 11, 96, stride = 4)
	norm1 = L.LRN(conv1, local_size = 5, alpha = 1e-4, beta = 0.75)
	pool1 = max_pool(norm1, 3, stride = 2)
	conv2, relu2 = conv_relu(pool1, 5, 256, pad = 2, group = 2)
	norm2 = L.LRN(conv2, local_size = 5, alpha = 1e-4, beta = 0.75)
	pool2 = max_pool(norm2, 3, stride = 2)
	conv3, relu3 = conv_relu(pool2, 3, 384, pad = 1)
	conv4, relu4 = conv_relu(relu3, 3, 384, pad = 1, group = 2)
	conv5, relu5 = conv_relu(relu4, 3, 256, pad = 1, group = 2)
	pool5 = max_pool(relu5, 3, stride = 2)
	fc6, relu6 = fc_relu(pool5, 4096)
	drop6 = L.Dropout(relu6, in_place = True)
	fc7, relu7 = fc_relu(drop6, 4096)
	drop7 = L.Dropout(relu7, in_place = True)
	fc8_f = L.InnerProduct(drop7, num_output = 10)
	
	loss = L.SigmoidCrossEntripyLoss(fc8_f, label)#Loss function can change whatever you need.

	if include_acc:
		acc = L.Accuracy(fc8_f, label)
		return to_proto(loss, acc)
	else:
		return to_proto(loss)


def make_alexnet_ava():
	with open('/home/u514/DTask/data/AVA/CNN_proto/ava_alex_CEL_train.prototxt', 'w') as f:
		print >> f, alexnet_ava_finetuning('/home/u514/DTask/data/AVA/ava_train.txt', batch_size = 100, include_acc = False)
		
	with open('/home/u514/DTask/data/AVA/CNN_proto/ava_alex_CEL_test.prototxt', 'w') as f:
		print >> f, alexnet_ava_finetuning('/home/u514/DTask/data/AVA/ava_test.txt', batch_size = 100, include_acc = False)



if __name__ == '__main__':
	make_alexnet_ava()

