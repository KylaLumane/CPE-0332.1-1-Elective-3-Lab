% Global Image thresholding using Otsu's method
% Load image
img = imread('flower.jpg'); % Replace 'image.jpg' with the correct image path

% Convert image to grayscale if it is not already
if size(img, 3) == 3
    img = rgb2gray(img);
end

% Calculate threshold using graythresh
level = graythresh(img);

% Convert into binary image using the computed threshold
bw = imbinarize(img, level);

% Display the original image and the binary image
figure(1); 
imshowpair(img, bw, 'montage'); 
title('Original Image (left) and Binary Image (right)');

% Multi-level thresholding using Otsu's method
% Calculate multiple thresholds using multithresh
levels = multithresh(img);

% Segment the image into regions using the imquantize function
seg_img = imquantize(img, levels);

% Display the original image and the segmented image
figure(2); 
imshowpair(img, seg_img, 'montage'); 
title('Original Image (left) and Segmented Image (right)');

% Global histogram threshold using Otsu's method
% Calculate a 16-bin histogram for the image
[counts, x] = imhist(img, 16);
figure(3);
stem(x, counts);
title('Histogram');

% Compute a global threshold using the histogram counts
T = otsuthresh(counts);

% Create a binary image using the computed threshold and display the image
bw = imbinarize(img, T);
figure(4); 
imshow(bw); 
title('Binary Image');

% Region-based segmentation using K means clustering
img2 = imread('flower.jpg'); % Replace with the correct image path

% Convert the image to grayscale
bw_img2 = im2gray(img2);
figure(5);
imshow(bw_img2);
title('Grayscale Image');

% Segment the image into three regions using k-means clustering
[L, centers] = imsegkmeans(bw_img2, 3);
B = labeloverlay(bw_img2, L);
figure(6); 
imshow(B); 
title('Labeled Image');

% Connected-component labeling
% Convert the image into binary
bin_img2 = imbinarize(bw_img2);

% Label the connected components
[labeledImage, numberOfComponents] = bwlabel(bin_img2);

% Display the number of connected components
disp(['Number of connected components: ', num2str(numberOfComponents)]);

% Assign a different color to each connected component
coloredLabels = label2rgb(labeledImage, 'hsv', 'k', 'shuffle');

% Display the labeled image
figure(7);
imshow(coloredLabels); 
title('Labeled Image');

% Adding noise to the image then segmenting it using Otsu's method
img_noise = imnoise(img, 'salt & pepper', 0.09);

% Calculate single threshold using multithresh
level_noise = multithresh(img_noise);

% Segment the image into two regions using the imquantize function
seg_img_noise = imquantize(img_noise, level_noise);

% Display the original image and the segmented image
figure(8); 
imshowpair(img_noise, seg_img_noise, 'montage'); 
title('Original Image (left) and Segmented Image with noise (right)');

% Segment the image into two regions using k-means clustering
RGB = imread('flower.jpg'); % Replace with the correct image path
L = imsegkmeans(RGB, 2); 
B = labeloverlay(RGB, L);
figure(9); 
imshow(B); 
title('Labeled Image');

% Create a set of 24 Gabor filters, covering 6 wavelengths and 4 orientations
wavelength = 2.^(0:5) * 3;
orientation = 0:45:135;
g = gabor(wavelength, orientation);

% Convert the image to grayscale
bw_RGB = im2gray(im2single(RGB));

% Filter the grayscale image using the Gabor filters and display the filtered images in a montage
gabormag = imgaborfilt(bw_RGB, g); 
figure(10); 
montage(gabormag, "Size", [4 6]);

% Smooth each filtered image to remove local variations
for i = 1:length(g)
    sigma = 0.5 * g(i).Wavelength;
    gabormag(:,:,i) = imgaussfilt(gabormag(:,:,i), 3 * sigma);
end
figure(11); 
montage(gabormag, "Size", [4 6]);

% Get the x and y coordinates of all pixels in the input image
nrows = size(RGB, 1);
ncols = size(RGB, 2);
[X, Y] = meshgrid(1:ncols, 1:nrows); 
featureSet = cat(3, bw_RGB, gabormag, X, Y);

% Segment the image into two regions using k-means clustering with the supplemented feature set
L2 = imsegkmeans(featureSet, 2, "NormalizeInput", true); 
C = labeloverlay(RGB, L2);
figure(12);
imshow(C); 
title("Labeled Image with Additional Pixel Information");
