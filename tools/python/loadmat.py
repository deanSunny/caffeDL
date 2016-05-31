'''
@author: Dean
@CAMALAB
2016-5-16
'''
#import 
import os
import numpy as np
import h5py

class LoadMatData():
	'''
	This class is used to loading *.mat file.
	The module of h5py is necessary for this class, scipy.io do not support *.mat > --version 7.3 .
	'''
	def __init__(self, matfile):
		self.matfile = matfile
		self.check_mat()
	
	def load_cell(self, matind):
		'''
		Load cell type. 
		'''
		self.matind = matind
		raw_data_file = h5py.File(self.matfile, 'r')
		raw_data = [raw_data_file[elem[0]][:] for elem in raw_data_file[self.matind]]
		raw_data_t = []
		for ind in xrange(len(raw_data)):
			raw_data_t.append(np.transpose(raw_data[ind]))
		raw_data_file.close()

		return raw_data_t

	def load_matrix(self, matind):
		'''
		Load matrix type.
		'''
		self.matind = matind
		raw_data_file =  h5py.File(self.matfile, 'r')
		raw_data = raw_data_file[self.matind][:]
		raw_data_file.close()
		
		raw_data = np.transpose(raw_data)

		return raw_data

	def transformer_selective_search(self, boxes):
		'''
		The boxes must matrix.
		'''
		self.mat = boxes
		box_list = []
		for i in xrange(len(self.mat)):
			box_list.append(self.mat[i][:, (1, 0, 3, 2)] - 1)

	def check_mat(self):
		assert os.path.exists(self.matfile), \
				'Mat file not found as: {}'.format(self.matfile)
		

if __name__ == "__main__":
	mat_path = '/home/u514/DTask/sunnyMaster/matlab/Pedestrian/bboxes'
	mat_mat = os.path.join(mat_path, 'selective_search_data_test.mat')
	mat_cell = os.path.join(mat_path, 'selective_search_data_train.mat')
	mat_name = 'boxes'
	load_mat = LoadMatData(mat_cell)
	load_mat.load_cell(mat_name)
	#print load_mat.load_matrix(mat_name)
