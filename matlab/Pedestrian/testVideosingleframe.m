function [ output_args ] = testVideosingleframe( video_path )
%testVideosingleframe : test video into fast-rcnn
%   @author: Dean
%   @CAMALAB
%   2016-5-18
%   input: video_path, the path of videos (e.g., avi, mp4)
%%
    root_path = fullfile('/home', 'u514', 'fast-rcnn');
    caffe_path = fullfile(root_path, 'caffe-fast-rcnn', 'matlab', 'caffe');
    addpath(caffe_path);
    use_gpu = true;
    def = fullfile(root_path, 'models', 'CaffeNet', 'test_camalab.prototxt');
    net = fullfile(root_path, 'output', 'default', 'caffe_net_camalab_pedestrian',...
                        'caffenet_fast_rcnn_camalab_iter_100000.caffemodel');
    model = fast_rcnn_load_net(def, net, use_gpu);
    
    Classes = {'person'};
    personid = 1;
%     addpath(genpath(fullfile(root_path, 'matlab', 'SelectiveSearchCodeIJCV')));
%     video = VideoReader(video_path);
%     nFrame = video.NumberOfFrames;
    imagefile = dir(video_path);
    for curFrame = 300 : length(imagefile)
%         im = read(video, curFrame);
        im = imread(fullfile(video_path, imagefile(curFrame).name));
        demo(model, im, personid, {'person'});
        msg = [num2str(curFrame), ' frame done.'];
        disp(msg);
        return;
    end
end
%%
function demo(model, im, cls_inds, cls_names)
    im = imresize(im, [480, 640]);
    tic;
    boxes = selective_search_boxes(im);
    boxes = boxes(:, [2,1,4,3]) - 1;
    boxes = single(boxes);
    toc;
    disp('OP done.');
    dets = fast_rcnn_im_detect(model, im, boxes);
    THRESH = 0.5;
    
    for j = 1:length(cls_inds)
      cls_ind = cls_inds(j);
      cls_name = cls_names{j};
      I = find(dets{cls_ind}(:, end) >= THRESH);
      showboxes_1(im, dets{cls_ind}(I, :));
      title(sprintf('%s detections with p(%s | box) >= %.3f', ...
                    cls_name, cls_name, THRESH))
      %hold on;
      %fprintf('\n> Press any key to continue');
%       pause;
    end
end

