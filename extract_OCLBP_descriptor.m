function descs = extract_OCLBP_descriptor(imageFile,kpoints,P,R,flag)
% Author: Chao Zhu (LIRIS, Ecole Centrale de Lyon)
% Email: chao.zhu@ec-lyon.fr
%
% This function extracts the orthogonal combination of local binary
% patterns (OC-LBP) descriptors for each keypoint in the input image.
%
% Reference:
% C. Zhu, C.-E. Bichot and L. Chen, Image region description using orthogonal
% combination of local binary patterns enhanced with color information,
% Pattern Recognition (2013), http://dx.doi.org/10.1016/j.patcog.2013.01.003
%
% Acknowledgment:
% I would like to thank Machine Vision Group, University of Oulu for
% sharing their code of LBP, which is used in this function.
%
% Input:
%     imageFile -- name of input image (gray or color)
%     kpoints   -- position of keypoints (2*N matrix with X coordinates in
%                  the first row and Y coordinates in the second row, N: number
%                  of keypoints)
%     P         -- number of neighboring pixels in OC-LBP (must be times of 4)
%     R         -- radius of neighboring circle in OC-LBP
%            common (P,R) pairs are: (8,1),(12,2),(16,2),(20,3),(24,3),etc.
%     flag      -- color space to compute OC-LBP (including: gray, rgb, nrgb,
%                  opponent, nopponent, hsv)
% Output:
%     descs     -- final OC-LBP descriptors (M*N matrix, M: dimension of
%                  descriptor, N: number of keypoints)

% Check input arguments
if rem(P,4) ~= 0
    fprintf('Error: number of neighboring pixels must be times of 4!\n');
    return;
end

% Set parameters
N = 4;           % number of neighboring points in each group
D = 2^N;         % dimension of LBP histogram
O = 20;          % coordinate offset according to keypoint for square patch
M = 3;           % M by M cells for each square patch

% Read image data
im = imread(imageFile);
[Hei,Wid,dim] = size(im);

% Transform image data according to color space
if strcmp(flag,'gray')
    num_channel = 1;
    if dim > 1
        im = rgb2gray(im);
    end
else
    num_channel = 3;
    if dim == 1
        fprintf('Error: not a color image!\n');
        return;
    end
    
    switch flag
        case 'rgb'
        case 'nrgb'
            im = im2double(im);
            
            im_1 = im(:,:,1);
            im_2 = im(:,:,2);
            im_3 = im(:,:,3);
            
            ALL = im_1+im_2+im_3;
            mask = ALL>0;
            
            r = zeros(Hei,Wid);
            r(mask) = im_1(mask)./ALL(mask);
            r(~mask)=1/3;
            
            g = zeros(Hei,Wid);
            g(mask) = im_2(mask)./ALL(mask);
            g(~mask)=1/3;
            
            b = zeros(Hei,Wid);
            b(mask) = im_3(mask)./ALL(mask);
            b(~mask)=1/3;
            
            im = cat(3,r,g,b);
        case 'opponent'
            im = im2double(im);
            
            im_1 = im(:,:,1);
            im_2 = im(:,:,2);
            im_3 = im(:,:,3);
            
            O1 = (im_1-im_2)/sqrt(2);
            O2 = (im_1+im_2-2*im_3)/sqrt(6);
            O3 = (im_1+im_2+im_3)/sqrt(3);
            
            im = cat(3,O1,O2,O3);
        case 'nopponent'
            im = im2double(im);
            
            im_1 = im(:,:,1);
            im_2 = im(:,:,2);
            im_3 = im(:,:,3);
            
            O1 = (im_1-im_2)/sqrt(2);
            O2 = (im_1+im_2-2*im_3)/sqrt(6);
            O3 = (im_1+im_2+im_3)/sqrt(3);
            
            mask = O3>0;
            
            CH1 = zeros(Hei,Wid);
            CH1(mask) = O1(mask)./O3(mask);
            CH1(~mask) = 0;
            
            CH2 = zeros(Hei,Wid);
            CH2(mask) = O2(mask)./O3(mask);
            CH2(~mask) = 0;
            
            im = cat(3,CH1,CH2,O3);
        case 'hsv'
            im = rgb2hsv(im);
        otherwise
            fprintf('Error: unknown color space!\n');
            return;
    end
end

% Calculate neighboring points
npoints = zeros(P,2);
a = 2*pi/P;

for i = 1:P
    npoints(i,1) = -R*sin((i-1)*a);
    npoints(i,2) = R*cos((i-1)*a);
end

% Divide neighboring points into orthogonal groups and calculate final
% OC-LBP descriptors
num_group = P/4;
descs = cell(num_channel*num_group,1);

for j = 1:num_channel
    for i = 1:num_group
        seq_group = npoints(i:num_group:P,:);
        % Compute LBP code for each pixel of the image
        lbp_image = lbp(im(:,:,j),seq_group,0,'i');
        % Calculate OC-LBP descriptor for each keypoint
        descs{(j-1)*num_group+i} = mex_computeLBPonKeypoints(lbp_image,kpoints,D,O,M);
    end
end

descs = cell2mat(descs);
descs = round(descs*512);
