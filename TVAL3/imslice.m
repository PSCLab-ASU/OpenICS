% imslice.m
% 
% Slices an image into submatrices, ordered into a cell array.
% 
% Usage: slices = imslice(img,img_channel,img_width,img_height,size)
%
% img - The image to slice into submatrices.
%
% img_channel - The number of channels in the image.
%
% img_width - The width of the image.
%
% img_height - The height of the image.
%
% slice_size - The size of each slice. Can be a vector with 2 elements or
%              an integer. Vector is ordered [width, height].
%

function slices = imslice(img,img_channel,img_width,img_height,slice_size)

    if img_channel > 1
        error('ERROR: Multi-channel images not yet supported!');
    end
    
    if any(mod([img_width, img_height], slice_size))
        error('ERROR: Slice dimensions do not match image dimensions!');
    end

    if numel(slice_size) == 1
        slices = mat2cell(img, repelem(slice_size, img_height / slice_size), repelem(slice_size, img_width / slice_size));
    elseif numel(slice_size) == 2
        slices = mat2cell(img, repelem(slice_size(2), img_height / slice_size(2)), repelem(slice_size(1), img_width / slice_size(1)));
    else
        error('ERROR: Size vector larger than 2 dimensions!');
    end

end
