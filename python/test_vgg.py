'''
@author: Dean
@CAMALAB	
@2016-7-22
'''
#import 
import os, sys
import config
import caffe
import numpy as np

class test_net(object):
	def __init__(self, model_def, caffe_model, gpu_id):
		self.model_def = model_def
		self.caffe_model = caffe_model
		#self.mean_file = mean_file
		#caffe.set_mode_cpu()
		caffe.set_mode_gpu()
		caffe.set_device(int(gpu_id))
		self.net = caffe.Net(self.model_def, self.caffe_model, caffe.TEST)

		self.transformer = caffe.io.Transformer({'data': 
			self.net.blobs['data'].data.shape})
		#self.transformer.set_mean('data', np.load(self.mean_file).mean(1).mean(1))
		self.transformer.set_transpose('data', (2, 0, 1))
		self.transformer.set_raw_scale('data', 255.0)
		self.transformer.set_channel_swap('data', (2, 1, 0))

		self.net.blobs['data'].reshape(1, 3, 224, 224)
		
	def test_model(self, im_path):
		im = caffe.io.load_image(im_path)
		im = self.transformer.preprocess('data', im)
		#prediction = self.net.predict(im)
		self.net.blobs['data'].data[...] = im
		self.net.forward()
		prediction = self.net.blobs['prob'].data
		return prediction

def get_image_index_from_txt(filename):
	label = []
	image_ind = []
	with open(filename, 'r') as f:
		line = f.readline().strip()
		while line:
			temp_line = line.split(' ')
			image_ind.append(temp_line[0])
			label.append(temp_line[1])
			line = f.readline().strip()
	return image_ind, label

if __name__ == "__main__":
	im_path = '/home/u514/DTask/data/IntestineHemorrhage/imgdataAugu/img6911_96743.jpg'
	out_dir = '/home/u514/DTask/data/IntestineHemorrhage/prediction'
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
	
	#pre = tn.test_model(im_path)
	test_ind = range(1, 11)
	for tid in test_ind:
		if tid < 10:
			continue
		model_def = '/home/u514/DTask/data/IntestineHemorrhage/models/vgg_16_deploy.prototxt'
		caffe_model = '/home/u514/DTask/data/IntestineHemorrhage/results/{}/vgg16_{}_inter_5000.caffemodel'.format(tid, tid)
		tn = test_net(model_def, caffe_model, 3)
		test_root = '/home/u514/DTask/data/IntestineHemorrhage/label'
		test_file = os.path.join(test_root, 'test{}.txt'.format(tid))
		image_ind, label = get_image_index_from_txt(test_file)
	
		pre_file = os.path.join(out_dir, 'test{}.txt'.format(tid))

		with open(pre_file, 'a') as f:
			for i, im in enumerate(image_ind):
				pre = tn.test_model(im)
				f.write('{:f} {:f} {:s}\n'.format(pre[0][0], pre[0][1], label[i]))
				print '{} Done.'.format(im.split('/')[-1])
	


