function testVideo( video_path )
%TESTVIDEO : test video into fast-rcnn
%   @author: Dean
%   @CAMALAB
%   2016-5-18
%   input: video_path, the path of videos (e.g., avi, mp4)
%%
    root_path = fullfile('/home', 'u514', 'fast-rcnn');
    tool_path = '/home/u514/fast-rcnn/matlab/';
    caffe_path = fullfile(root_path, 'caffe-fast-rcnn', 'matlab', 'caffe');
    addpath(caffe_path, tool_path);
    use_gpu = true;
    def = fullfile(root_path, 'models', 'CaffeNet', 'test_camalab.prototxt');
    net = fullfile(root_path, 'output', 'default', 'caffe_net_camalab_pedestrian',...
                        'caffenet_fast_rcnn_camalab_iter_100000.caffemodel');
    model = fast_rcnn_load_net(def, net, use_gpu);
    
    Classes = {'person'};
    personid = 1;
%     addpath(genpath(fullfile(root_path, 'matlab', 'SelectiveSearchCodeIJCV')));
    video = VideoReader(video_path);
    nFrame = video.NumberOfFrames;
    for curFrame = 1 : nFrame
        im = read(video, curFrame);
        demo(model, im, personid, {'person'});
        msg = [curFrame, ' frame done.'];
        disp(msg);
    end
end

function demo(model, im, cls_inds, cls_names)
    im = imresize(im, [480, 640]);
    tic;
    boxes = selective_search_boxes(im);
    boxes = boxes([2,1,4,3]) - 1;
    boxes = single(boxes);
    toc;
    disp('OP done.');
    dets = fast_rcnn_im_detect(model, im, boxes);
    THRESH = 0.5;
    
    for j = 1:length(cls_inds)
      cls_ind = cls_inds(j);
      cls_name = cls_names{j};
      I = find(dets{cls_ind}(:, end) >= THRESH);
      showboxes(im, dets{cls_ind}(I, :));
      title(sprintf('%s detections with p(%s | box) >= %.3f', ...
                    cls_name, cls_name, THRESH))
      fprintf('\n> Press any key to continue');
%       pause;
    end
end