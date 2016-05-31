% @author: Dean
% @CAMALAB
% 2016-5-9
%%
addpath('..\\');
addpath('anno_vbbtxt');
vbbstxt_dir = dir('anno_vbbtxt');  
bboxes = {};
for i = 3 : length(vbbstxt_dir)
    vbbstxt_name = vbbstxt_dir(i).name;
    vbbstxt_obj = vbb('vbbLoadTxt', ['anno_vbbtxt\\', vbbstxt_name]);
%     bbs_num = vbb('numObj', vbbstxt_obj);
%     bboxes = {};
%     for bbsn = 1 : bbs_num 
% %         bbs_obj = vbb('get', vbbstxt_obj, bbsn, [1], [vbbstxt_obj.nFrame]);
% %         bbs_obj = vbb('get', vbbstxt_obj, bbsn);
% %         bboxes{bbsn} = bbs_obj.pos;
%         
%     end
    for nFrame = 1 : vbbstxt_obj.nFrame
        [bboxes{nFrame,i -2 },posv,lbls] = vbb( 'frameAnn', vbbstxt_obj, nFrame, 'person');
    end
end
save_path = fullfile('bboxes', 'bboxes_002-003.mat');
save(save_path, 'bboxes');