function getTestOP( impath )
%    @author: Dean
%    @CAMALAB
%    2016-5-11
%GETTESTOP:  
%   im: The image you want to extract  bounding boxes;
im = imread(impath);
boxes = selective_search_boxes(im);
boxsavefile = fullfile('bboxes', 'selective_search_data_test.mat');
imageListfile = fullfile('labels', 'imageName_test.mat');
save(boxsavefile, 'boxes');
save(imageListfile, 'impath');
end
