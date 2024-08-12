
img = imread('flower.jpg'); % Replace with the correct image path

img_noise = imnoise(grayImg, 'salt & pepper', 0.09);
level = multithresh(img_noise);
seg_img = imquantize(img_noise, level) ;

figure(6);
imshowpair (img_noise, seg_img, 'montage');
title('Original Image (left) and Segmented Image with noise (right) ');

RGB = imread('flower.jpg');
L= imsegkmeans(RGB,2); B = labeloverlay(RGB,L) ;

figure(7);
imshow(B) ;
title('Labeled Image');

wavelength = 2.^(0:5) * 3;
orientation = 0:45:135;
g = gabor (wavelength, orientation) ;
bw_RGB = im2gray(im2single(RGB));
gabormag = imgaborfilt (bw_RGB,g) ;

figure(8);
montage (gabormag, "Size", [4 6])

    for i = 1:1ength (g)
        sigma = 0.5*g(i) .Wavelength;
        gabormag(:,:,i) = imgaussfilt(gabormag(:,:,i), 3*sigma) ;

    end

figure(9);
montage (gabormag, "Size", [4 6])

nrows = size(RGB, 1) ;
ncols = size(RGB,2);

[X,Y] = meshgrid(1:ncols,1:nrows) ;