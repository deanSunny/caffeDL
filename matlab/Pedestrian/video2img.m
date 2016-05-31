function video2img()
%VIDEO2IMG : Translate video files to images.
%   @author: Dean
%   @CAMALAB
%   2016-5-9
%%
libpath = '.\';
addpath(libpath);

videoname = {'002.mp4', '003.mp4'};
for nVideo = 1 : length(videoname)
    [v, nFrame] = loadVideo(videoname{nVideo});
    savepath = fullfile('images', videoname{nVideo});
        if ~ exist(savepath, 'dir')
            mkdir(savepath);
        end
    addpath(savepath);
    for nF = 1 : nFrame
        getFrame(v, nF, savepath);
%         imshow(im);
        
    end
end

function  getFrame( v , curentFrame,savepath)
% vFilename: video file name.
%%
im = read(v, curentFrame);
imwrite(im, fullfile(savepath, [num2str(curentFrame), '.jpg']), 'jpg')


function [v, nFrame]= loadVideo(vFilename)
if ~exist(vFilename, 'file')
    disp('No such file, please check it out.');
    return;
end
v = VideoReader(vFilename);
nFrame = v.NumberOfFrames;