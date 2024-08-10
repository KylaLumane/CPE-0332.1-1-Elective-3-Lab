import cv2
import numpy as np
from scipy.ndimage import convolve, gaussian_filter
from skimage import restoration
import matplotlib.pyplot as plt

# Read the image
img = cv2.imread('flower.jpg')

# Convert the image from BGR to RGB (since OpenCV reads in BGR format)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Display the original image
plt.figure(1)
plt.imshow(img)
plt.title('Original Image')
plt.axis('off')

# Convert to grayscale if the image is RGB
if len(img.shape) == 3:
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
else:
    img_gray = img

# Display the grayscale image
plt.figure(2)
plt.imshow(img_gray, cmap='gray')
plt.title('Grayscale')
plt.axis('off')

# Add blur to the image (motion blur)
len = 21
theta = 11
psf = np.zeros((len, len))
center = len // 2
for i in range(len):
    offset = int(np.round((i - center) * np.tan(np.radians(theta))))
    if 0 <= center + offset < len:
        psf[i, center + offset] = 1
psf /= psf.sum()

img_blur = convolve(img_gray, psf)

# Show the motion blurred image
plt.figure(3)
plt.imshow(img_blur, cmap='gray')
plt.title('Motion Blurred Image')
plt.axis('off')

# Gaussian filtering
img_gaussian_filtered = gaussian_filter(img_blur, sigma=1)

# Display the Gaussian filtered image
plt.figure(4)
plt.imshow(img_gaussian_filtered, cmap='gray')
plt.title('Filtered Image (Gaussian)')
plt.axis('off')

# Sharpening using unsharp masking
img_sharpened = cv2.addWeighted(img_blur, 1.5, gaussian_filter(img_blur, sigma=1), -0.5, 0)

# Display the sharpened image
plt.figure(5)
plt.imshow(img_sharpened, cmap='gray')
plt.title('Sharpened Image')
plt.axis('off')

# Add Gaussian noise and remove it using a median filter
img_noisy = img_gray + np.random.normal(0, 25, img_gray.shape).astype(np.uint8)
img_noisy = np.clip(img_noisy, 0, 255)
img_noisy_removed = cv2.medianBlur(img_noisy, 5)

# Display the noisy image
plt.figure(6)
plt.imshow(img_noisy, cmap='gray')
plt.title('Noisy')
plt.axis('off')

# Display the noise-removed image
plt.figure(7)
plt.imshow(img_noisy_removed, cmap='gray')
plt.title('Noise Removed')
plt.axis('off')

# Deblurring using Wiener filter
estimated_nsr = 0.01
img_deblurred = restoration.wiener(img_blur, psf, estimated_nsr)

# Display the deblurred image
plt.figure(8)
plt.imshow(img_deblurred, cmap='gray')
plt.title('Deblurred Image')
plt.axis('off')

plt.show()
