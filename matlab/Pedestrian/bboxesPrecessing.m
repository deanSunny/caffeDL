function [ new_bboxes ] = bboxesPrecessing( bbox)
%BBOXESPRECESSING make bbox change to [x1, y1, x2, y2].
%  @author: Dean
%  @CAMALAB
%  2016-5-10
%  bboxes: file type -> *.mat
%%
new_bboxes = {};
load(bbox);
[nI, nF]= size(bboxes);
for bF = 1 : nF
    for bI = 1: nI
        bbox_one = bboxes{bI, bF};
        if ~isempty(bbox_one)
            bbox_one(:, 5) = [];
            bbox_one(:, 3) = bbox_one(:, 1) + bbox_one(:, 3);
            bbox_one(:, 4) = bbox_one(:, 2) + bbox_one(:, 4);   
        end
        new_bboxes{bI, bF} = bbox_one;
    end
end
end

