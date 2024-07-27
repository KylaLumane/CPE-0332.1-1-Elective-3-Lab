
img_color = imread('orange.png');

Red = img_color;

%Red
Red(:,:,2) = 0;
Red(:,:,3) = 0;
figure(1);
imshow(Red);
whos Red;

%Green
Green = img_color;
Green(:, :, 1) = 0;
Green(:, :, 3) = 0;
figure(2);
imshow(Green);
whos Green;

%Blue
Blue = img_color;
Blue(:, :, 1) = 0;
Blue(:, :, 2) = 0; 
figure(3);
imshow(Blue);
whos Blue;

%Grayscaled
Grayscaled = rgb2gray(img_color);
figure(4);
imshow(Grayscaled);
whos Grayscaled;

imwrite(img_color,'orange.png')
imwrite(Red,'red_img.png');
imwrite(Green,'green_img.png');
imwrite(Blue,'blue_img.png');
imwrite(Grayscaled,'gray_img.png');

imwrite(img_color, 'orange.png', 'jpg', 'Quality', 100)
imwrite(Red, 'red_img.png', 'jpg', 'Quality', 100)
imwrite(Green, 'green_img.png', 'jpg', 'Quality', 100)
imwrite(Blue, 'blue_img.png', 'jpg', 'Quality', 100)
imwrite(Grayscaled, 'gray_img.png', 'jpg', 'Quality', 100) 