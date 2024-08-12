import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import color, filters, measure, segmentation, util
from skimage.filters import threshold_multiotsu, gabor_kernel
from sklearn.cluster import KMeans

# Load image
img = cv2.imread('flower.jpg')  # Replace 'flower.jpg' with the correct image path
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Global Image thresholding using Otsu's method
# Calculate threshold using Otsu's method
_, bw = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Display the original image and the binary image
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_gray, cmap='gray')
plt.title('Original Image')
plt.subplot(1, 2, 2)
plt.imshow(bw, cmap='gray')
plt.title('Binary Image')
plt.show()

# Multi-level thresholding using Otsu's method
levels = threshold_multiotsu(img_gray, classes=3)
seg_img = np.digitize(img_gray, bins=levels)

# Display the original image and the segmented image
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_gray, cmap='gray')
plt.title('Original Image')
plt.subplot(1, 2, 2)
plt.imshow(seg_img, cmap='gray')
plt.title('Segmented Image')
plt.show()

# Global histogram threshold using Otsu's method
# Calculate a 16-bin histogram for the image
counts, x = np.histogram(img_gray, bins=16)
plt.figure()
plt.stem(x[:-1], counts, basefmt=" ")
plt.title('Histogram')
plt.show()

# Compute a global threshold using the histogram counts
T = filters.threshold_otsu(img_gray)
bw = img_gray > T

# Create a binary image using the computed threshold and display the image
plt.figure()
plt.imshow(bw, cmap='gray')
plt.title('Binary Image')
plt.show()

# Region-based segmentation using K means clustering
img2 = cv2.imread('flower.jpg')  # Replace with the correct image path
bw_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# K-means clustering
kmeans = KMeans(n_clusters=3, random_state=0).fit(bw_img2.reshape(-1, 1))
L = kmeans.labels_.reshape(bw_img2.shape)
B = color.label2rgb(L, image=bw_img2, bg_label=0)

plt.figure()
plt.imshow(B)
plt.title('Labeled Image')
plt.show()

# Connected-component labeling
bin_img2 = bw_img2 > filters.threshold_otsu(bw_img2)
labeledImage = measure.label(bin_img2)
numberOfComponents = labeledImage.max()

# Display the number of connected components
print(f'Number of connected components: {numberOfComponents}')

# Assign a different color to each connected component
coloredLabels = color.label2rgb(labeledImage, bg_label=0)

# Display the labeled image
plt.figure()
plt.imshow(coloredLabels)
plt.title('Labeled Image')
plt.show()

# Adding noise to the image then segmenting it using Otsu's method
img_noise = util.random_noise(img_gray, mode='s&p', amount=0.09)
img_noise = (img_noise * 255).astype(np.uint8)

# Multi-level thresholding using Otsu's method
levels_noise = threshold_multiotsu(img_noise, classes=3)
seg_img_noise = np.digitize(img_noise, bins=levels_noise)

# Display the original image and the segmented image
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_noise, cmap='gray')
plt.title('Original Image with Noise')
plt.subplot(1, 2, 2)
plt.imshow(seg_img_noise, cmap='gray')
plt.title('Segmented Image with Noise')
plt.show()

# Segment the image into two regions using k-means clustering
RGB = cv2.imread('flower.jpg')  # Replace with the correct image path
RGB_gray = cv2.cvtColor(RGB, cv2.COLOR_BGR2GRAY)
kmeans = KMeans(n_clusters=2, random_state=0).fit(RGB_gray.reshape(-1, 1))
L = kmeans.labels_.reshape(RGB_gray.shape)
B = color.label2rgb(L, image=RGB_gray, bg_label=0)

plt.figure()
plt.imshow(B)
plt.title('Labeled Image')
plt.show()

# Create a set of 24 Gabor filters, covering 6 wavelengths and 4 orientations
wavelengths = np.array([3, 6, 12, 24, 48, 96])
orientations = np.array([0, 45, 90, 135])
gabor_filters = [gabor_kernel(wavelength, theta=np.deg2rad(orientation))
                 for wavelength in wavelengths for orientation in orientations]

# Filter the grayscale image using the Gabor filters and display the filtered images in a montage
gabormag = [cv2.filter2D(RGB_gray, cv2.CV_8UC3, np.real(gabor_filter)) for gabor_filter in gabor_filters]
gabormag = np.array(gabormag)
gabormag = np.transpose(gabormag, (1, 2, 0))

plt.figure(figsize=(12, 8))
for i in range(len(gabor_filters)):
    plt.subplot(4, 6, i+1)
    plt.imshow(gabormag[:, :, i], cmap='gray')
    plt.axis('off')
plt.suptitle('Filtered Images using Gabor Filters')
plt.show()

# Smooth each filtered image to remove local variations
gabormag_smoothed = np.array([cv2.GaussianBlur(gabormag[:, :, i], (0, 0), sigmaX=3*0.5*wavelengths[i//4]) for i in range(len(gabor_filters))])
gabormag_smoothed = np.transpose(gabormag_smoothed, (1, 2, 0))

plt.figure(figsize=(12, 8))
for i in range(len(gabor_filters)):
    plt.subplot(4, 6, i+1)
    plt.imshow(gabormag_smoothed[:, :, i], cmap='gray')
    plt.axis('off')
plt.suptitle('Smoothed Gabor Filtered Images')
plt.show()

# Get the x and y coordinates of all pixels in the input image
nrows, ncols = RGB_gray.shape
X, Y = np.meshgrid(np.arange(ncols), np.arange(nrows))
feature_set = np.dstack((RGB_gray, gabormag_smoothed, X, Y))

# Segment the image into two regions using k-means clustering with the supplemented feature set
kmeans = KMeans(n_clusters=2, random_state=0).fit(feature_set.reshape(-1, feature_set.shape[2]))
L2 = kmeans.labels_.reshape(RGB_gray.shape)
C = color.label2rgb(L2, image=RGB, bg_label=0)

plt.figure()
plt.imshow(C)
plt.title('Labeled Image with Additional Pixel Information')
plt.show()
