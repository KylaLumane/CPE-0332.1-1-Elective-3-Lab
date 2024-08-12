import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from skimage import color, filters

# Load the image
img = cv2.imread('flower.jpg')  # MATLAB: imread('flower.jpg')
# Convert to grayscale if the image is BGR
if img.shape[2] == 3:
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # MATLAB: rgb2gray(img)
else:
    grayImg = img

# Adding noise to the image then segmenting it using Otsu's method
noiseImg = np.clip(grayImg + np.random.normal(0, 25, grayImg.shape), 0, 255).astype(np.uint8)
otsuThresh = filters.threshold_otsu(noiseImg)
segImgNoise = (noiseImg > otsuThresh).astype(np.uint8) * 255

plt.figure(6)
plt.subplot(1, 2, 1)
plt.imshow(noiseImg, cmap='gray')
plt.title('Noisy Image')
plt.subplot(1, 2, 2)
plt.imshow(segImgNoise, cmap='gray')
plt.title('Segmented Image with noise')
plt.show()

# Segmenting the image into two regions using K-Means clustering
RGB = cv2.imread('flower.jpg')
RGB = cv2.cvtColor(RGB, cv2.COLOR_BGR2RGB)
kMeans = KMeans(n_clusters=2, random_state=0).fit(RGB.reshape(-1, 3))
labels = kMeans.labels_.reshape(RGB.shape[:2])
labelOverlay = cv2.applyColorMap(np.uint8(labels * 255 / (labels.max() if labels.max() > 0 else 1)), cv2.COLORMAP_JET)

plt.figure(7)
plt.imshow(labelOverlay)
plt.title('Labeled Image')
plt.show()

# Creating and Applying Gabor Filters
def gaborFilter(img, wavelength, orientation):
    filters = []
    for theta in orientation:
        theta = np.deg2rad(theta)
        for lambda_ in wavelength:
            kernel = cv2.getGaborKernel((31, 31), 4.0, theta, lambda_, 0.5, 0, cv2.CV_32F)
            filters.append(kernel)
    return filters

wavelength = [2 ** i * 3 for i in range(6)]
orientation = list(range(0, 180, 45))
gaborKernels = gaborFilter(grayImg, wavelength, orientation)

gaborMag = np.zeros_like(grayImg, dtype=np.float32)
for kernel in gaborKernels:
    filteredImg = cv2.filter2D(grayImg, cv2.CV_32F, kernel)
    gaborMag = np.maximum(gaborMag, np.abs(filteredImg))

plt.figure(8)
num_kernels = len(gaborKernels)
for i in range(num_kernels):
    plt.subplot(4, 6, i + 1)
    plt.imshow(cv2.filter2D(grayImg, cv2.CV_32F, gaborKernels[i]), cmap='gray')
plt.suptitle('Gabor Filtered Images')
plt.show()

# Smoothing Gabor Filtered Images
for i in range(num_kernels):
    sigma = 0.5 * wavelength[i % len(wavelength)]
    gaborMag = cv2.GaussianBlur(gaborMag, (0, 0), sigma)

plt.figure(9)
for i in range(num_kernels):
    plt.subplot(4, 6, i + 1)
    plt.imshow(gaborMag, cmap='gray')
plt.suptitle('Smoothed Gabor Filtered Images')
plt.show()

# Feature Set for Clustering
x, y = np.meshgrid(np.arange(grayImg.shape[1]), np.arange(grayImg.shape[0]))
featureSet = np.stack([grayImg, gaborMag, x, y], axis=-1)
featureSetReshaped = featureSet.reshape(-1, featureSet.shape[-1])

kMeans = KMeans(n_clusters=2, random_state=0).fit(featureSetReshaped)
labels = kMeans.labels_.reshape(grayImg.shape)
labelOverlay = color.label2rgb(labels, image=RGB, bg_label=0)

plt.figure(10)
plt.imshow(labelOverlay)
plt.title('Labeled Image with Additional Pixel Information')
plt.show()
