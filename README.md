# caffeDL


#### python

- classify-folder.py:
	- Classify the CNN feature to predict the result of classes.

- get_feature_f7.py:
	- Extract the features from CNN layers (default: f7).

- get_feature_f7_function.py:
    - Extract the features from CNN layers (function).

- temproal_datalayers.py:
	- python layer of data layers for two-stream.


#### matlab

- SelectiveSearch:
	- SelectiveSearch used to extract bounding-box.
- demo_ve.m:
	- A simple demo for fast-rcnn test.


#### tools

- binaryproto_to_npy.py
- create_label.py
- loadmat.py
	- Use hdf5 to load *.mat file
- savetxt_from_npy.py:
	- A simple tools used to change the date type from npy to txt(Only for martrix).	
- getlabel.py getlabel_v2.py:
	- A simple tools used to get labels of dataset(e.g., UCF-101).
- getAVALabel.py:
	- Split the image set to Train, Test and Validation set.Save as HDF5 data type.
- imagecroper.py:
	- Crop the images to the crop-dim.
