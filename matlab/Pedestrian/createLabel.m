function createLabel( folderName )
%CREATELABEL : Create label function.
%   @author: Dean
%   @CAMALAB
%   2016-5-10
%%
imageLabels = {};
if ~exist(folderName, 'dir')
    mes = 'No such dir,please check it out.';
    disp(mes);
    return;
end
bboxes_path = 'bboxes/bboxes_002-003.mat';
bboxes_file = bboxesPrecessing(bboxes_path);
[cDim, num] = size(bboxes_file);
dirFolder = dir(folderName);
for nFolder = 1 : num
    for nImage = 1: cDim
        if nImage > 835 && nFolder == 1
            break;
        else if nImage <= 835 && nFolder == 1
              imageLabels{nImage, nFolder} = fullfile(folderName, dirFolder(nImage+2).name);
        else
              imageLabels{nImage, nFolder} = fullfile(folderName, dirFolder(nImage+837).name); 
            end
        end
    end
end
mkdir('labels');
save(fullfile('labels', 'imageName.mat'), 'imageLabels');

labelFile = fullfile('labels', 'testLabel1.txt');
imageFile = fullfile('labels', 'ImageList.txt');
for n = 1: num
    for m = 1 : cDim 
        if ~ isempty(imageLabels{m, n})
            [nBbox, axis] = size(bboxes_file{m, n});
            if nBbox < 5
                fMsg = [imageLabels{m, n}, ' 4 ', getMatrix(bboxes_file{m, n}(1,:)), getMatrix(bboxes_file{m, n}(2,:)), getMatrix(bboxes_file{m, n}(3,:)), getMatrix(bboxes_file{m, n}(4,:))];
            else
                fMsg = [imageLabels{m, n}, ' 5 ', getMatrix(bboxes_file{m, n}(1,:)), getMatrix(bboxes_file{m, n}(2,:)), getMatrix(bboxes_file{m, n}(3,:)), getMatrix(bboxes_file{m, n}(4,:)), getMatrix(bboxes_file{m, n}(5,:))];
            end
            imageList = imageLabels{m ,n};
            saveLabel(fMsg, labelFile);
            saveLabel(imageList, imageFile);
        end
    end
end
end
%%
function saveLabel(msg, fName)
    fid = fopen(fName, 'a');
%     dlmwrite(fName, msg, '-append');
    fprintf(fid,'%s\r\n', msg);
    fclose(fid);
end

function [string] = getMatrix(M)
    string = [];
    Len = length(M);
    for i = 1 : Len
        string = [string, num2str(M(i)) , ' '];
    end
end
