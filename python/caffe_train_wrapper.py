'''
@author: Dean
@CAMALAB
@2016-7-18
'''
#import 
import sys, os
CAFFE_PATH = '/home/u514/caffe-i/caffe-master/caffe/python'
TOOLS_LIB_PATH = '/home/u514/DTask/sunnyMaster/tools/python'
sys.path.insert(0, CAFFE_PATH)
sys.path.insert(0, TOOLS_LIB_PATH)
import caffe
from caffe.proto import caffe_pb2
import google.protobuf as pb2
from timer import Timer

class TrainWrapper(object):
	def __init__(self, solver_prototxt, output_dir, exp_time, pretrained_model=None, gpu_id=1):
		caffe.set_mode_gpu()
		caffe.set_device(int(gpu_id))
		self.output_dir = output_dir
		self.exp_time = exp_time
		self.solver = caffe.SGDSolver(solver_prototxt)
		if pretrained_model is not None:
			print 'Loading pretrained model weights from {:s}'.format(pretrained_model)
			self.solver.net.copy_from(pretrained_model)

		self.solver_param = caffe_pb2.SolverParameter()
		with open(solver_prototxt, 'rt') as f:
			pb2.text_format.Merge(f.read(), self.solver_param)
	
	def snapshot(self):
		net = self.solver.net
		infix = '_{:d}'.format(self.exp_time)
		filename = (self.solver_param.snapshot_prefix + infix +
				'_inter_{}'.format(self.solver.iter) + '.caffemodel')
		if not os.path.exists(self.output_dir):
			os.makedirs(self.output_dir)
		filename = os.path.join(self.output_dir, filename)

		net.save(str(filename))
		print 'Wrote snapshot to: {:s}'.format(filename)
		return filename
	
	def train_model(self, max_iters, TRAIN_SNAPSHOT_ITERS=2000):
		print 'test_iter: {}'.format(self.solver_param.test_iter[0])
		last_snapshot_iter = -1
		timer = Timer()
		model_paths = []
		self.solver.net.forward()
		self.solver.test_nets[0].forward()
		while self.solver.iter < max_iters:
			timer.tic()
			self.solver.step(1)
			timer.toc()
			if self.solver.iter % (10 * int(self.solver_param.display)) == 0:
				print 'speed: {:.3f}s / iter'.format(timer.average_time)
			timer.tic()
			self.test_model()
			timer.toc()
			if self.solver.iter % TRAIN_SNAPSHOT_ITERS == 0:
				last_snapshot_iter = self.solver.iter
				model_paths.append(self.snapshot())

		if last_snapshot_iter != self.solver.iter:
			model_paths.append(self.snapshot())

		return model_paths

	def test_model(self):
		accuracy = -1
		if self.solver.iter % int(self.solver_param.test_iter[0]) == 0:
			print 'Testing... iters: {:d}'.format(self.solver.iter)	
			for it in xrange(int(self.solver_param.test_interval)):
				self.solver.test_nets[0].forward()
				accuracy += self.solver.test_nets[0].blobs['accuracy'].data
			accuracy /= int(self.solver_param.test_interval)
			print 'Test Accuracy: {}'.format(accuracy)

if __name__ == '__main__':
	#tw = TrainWrapper(os.path.join(CAFFE_PATH, '../models', 'vgg/vgg_2048', 'temproal_net_solver.prototxt'), './', 1)
	pass
