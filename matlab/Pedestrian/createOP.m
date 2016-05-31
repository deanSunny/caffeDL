addpath(genpath('../SelectiveSearch/SelectiveSearchCodeIJCV'));

fid = fopen(fullfile('labels', 'ImageList.txt'), 'r');
tag = 1;
boxes = {};
while ~feof(fid)
    fline = fgetl(fid);
    if ~isempty(fline);
        disp(fline);
        tic;
        im = imread(fline);
        boxes{tag} = selective_search_boxes(im);
        toc;
        tag = tag + 1;
    end
end
savefile = fullfile('bboxes', 'selective_search_data_train.mat');
save(savefile, 'boxes');
fclose(fid);