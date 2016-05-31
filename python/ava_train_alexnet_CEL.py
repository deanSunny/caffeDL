'''
@author: Dean
@CAMALAB
2016-5-10

'''
#import 
import os, sys
import numpy as np
CAFFE_HOME = '/home/u514/caffe-i/caffe-master/caffe/python'
#CAFFE_TOOLS_HOME = '/home/u514/DTask/sunnyMaster/tools/python'
sys.path.insert(0, CAFFE_HOME)
#sys.path.insert(0, CAFFE_TOOLS_HOME)
import caffe, h5py
from pylab import *

def init_net(gpu_id, solver_path, caffe_model):
	caffe.set_mode_gpu()
	caffe.set_device(int(gpu_id))
	#net = caffe.Net(model_def, caffemodel, caffe.TRAIN)
	solver = caffe.SGDSolver(solver_path)
	solver.net.copy_from(caffe_model)
	return solver

def accuracy_plcc(data, label):
	cls_num = len(label[0])
	num = len(label)
	data_k = np.zeros(num)
	label_k = np.zeros(num)
	
	for k in xrange(num):
		for cls in xrange(cls_num):
			data_k[k] += data[k][cls] * (cls + 1)
			label_k[k] += label[k][cls] * (cls + 1)
	mean_data = np.mean(data_k)
	mean_label = np.mean(label_k)

	plcc_u = np.sum((label_k - mean_label) * (data_k - mean_data))
	plcc_p_l = np.sum((label_k - mean_label) ** 2)
	plcc_p_d = np.sum((data_k - mean_data) ** 2)
	plcc = plcc_u / np.sqrt(plcc_p_l * plcc_p_d)

	return plcc


def disp_iter(it, disp_inter, item, status):
	if it % disp_inter == 0:
		print 'Iteration {} {}: loss = {}'.format(it, status, item)
	else:
		return

def train_net(solver, maxiter, test_interval_num, display_num):
	solver.net.forward()
	solver.test_nets[0].forward()
	solver.step(1)
	
	niter = int(maxiter) #max_iter
	test_interval = int(test_interval_num)
	display = int(display_num)
	train_loss = np.zeros(niter)
	test_acc = np.zeros(int(np.ceil(niter * 1.0 / test_interval)))
	print test_acc.shape
	
	for it in xrange(1, niter):
		solver.step(1)
		train_loss[it] = solver.net.blobs['loss'].data
		solver.test_nets[0].forward(start='data')
		disp_iter(it, display, train_loss[it], 'training')
		if it % test_interval == 0:
			data = solver.test_nets[0].blobs['test_prob'].data
			label = solver.test_nets[0].blobs['label'].data
			test_acc[it / test_interval] =  accuracy_plcc(data, label)
			print test_acc[it / test_interval]
	
	fig_, ax1 = subplots()
	ax2 = ax1.twinx()
	ax1.plot(arange(niter), train_loss)
	ax2.plot(test_interval * arange(len(test_acc)), test_acc, 'r')
	ax1.set_xlabel('Iteration')
	ax1.set_ylabel('Train Loss')
	ax2.set_ylabel('Test Accuracy')
	save_fig_path = '/home/u514/DTask/data/AVA/results/testAva_alexNet_SCEL.png'
	fig_.savefig(save_fig_path)
	print 'Results have saved in {}'.format(save_fig_path)
	
	#check result
	#print solver.test_nets[0].blobs['fc7'].data.shape
	#print '_________________', solver.test_nets[0].blobs['label'].data.shape

if __name__ == "__main__":
	solver_path = '/home/u514/DTask/data/AVA/CNN_proto/ava_alexnet_solver.prototxt'
	caffemodel = '/home/u514/caffe-i/caffe-master/caffe/models/bvlc_alexnet/bvlc_alexnet.caffemodel'
	train_net(init_net(3, solver_path, caffemodel), 100000, 100, 20)

