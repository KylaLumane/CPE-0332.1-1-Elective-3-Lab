import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

# Read the image
img = cv2.imread('flower.jpg')

# Convert the image from BGR to RGB (since OpenCV reads in BGR format)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convert to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# Gaussian filtering with experimented value
h_gaussian = gaussian_filter(img_gray, sigma=5)

# Display the Gaussian filtered image
plt.figure()
plt.imshow(h_gaussian, cmap='gray')
plt.title('Filtered Image with Experimented Value (Gaussian)')
plt.axis('off')

# Display the histogram of the Gaussian filtered image
plt.figure()
plt.hist(h_gaussian.ravel(), bins=256, range=(0, 256), density=True)
plt.title('Histogram of the Experimented Value (Gaussian Filtered)')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')

# Add Gaussian noise with experimented values
mean = 0
img_noisy_exp1 = img_gray + np.random.normal(mean, 0.2 * 255, img_gray.shape).astype(np.uint8)
img_noisy_exp2 = img_gray + np.random.normal(mean, 0.1 * 255, img_gray.shape).astype(np.uint8)

# Clip pixel values to be in the correct range
img_noisy_exp1 = np.clip(img_noisy_exp1, 0, 255)
img_noisy_exp2 = np.clip(img_noisy_exp2, 0, 255)

# Display the noisy images
plt.figure()
plt.imshow(img_noisy_exp1, cmap='gray')
plt.title('Noisy Using Experimented Value (Gaussian is 0.5)')
plt.axis('off')

plt.figure()
plt.imshow(img_noisy_exp2, cmap='gray')
plt.title('Noisy Using Experimented Value (Gaussian is 0.1)')
plt.axis('off')

# Display the histogram for noisy images
plt.figure()
plt.hist(img_noisy_exp1.ravel(), bins=256, range=(0, 256), density=True)
plt.title('Histogram of Noisy Image Experimented Value 1')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')

plt.figure()
plt.hist(img_noisy_exp2.ravel(), bins=256, range=(0, 256), density=True)
plt.title('Histogram of Noisy Image Experimented Value 2')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')

plt.show()
