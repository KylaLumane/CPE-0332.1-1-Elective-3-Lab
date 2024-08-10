import cv2

# Function to add border and label to the image
def add_border_and_label(img, label, border_size=50):
    # Add border to the image
    bordered_img = cv2.copyMakeBorder(img, border_size, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    # Add label to the border
    cv2.putText(bordered_img, label, (10, border_size - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return bordered_img

# Read the image
img = cv2.imread("flower.jpg")

# Rotate by 30 degrees
center_img = (img.shape[1] // 2, img.shape[0] // 2)
rotation_img = cv2.getRotationMatrix2D(center_img, 30, 1)
rotated_img = cv2.warpAffine(img, rotation_img, (img.shape[1], img.shape[0]))

# Flip horizontally
flipped_img = cv2.flip(rotated_img, 1)

# Add border and labels to the images
img_with_border = add_border_and_label(img, 'Original Image')
rotated_img_with_border = add_border_and_label(rotated_img, 'Rotated 30 DEG')
flipped_img_with_border = add_border_and_label(flipped_img, 'Rotated & Flipped')

# Display results
cv2.imshow('Original Image', img_with_border)
cv2.imshow('Rotated 30°', rotated_img_with_border)
cv2.imshow('Rotated & Flipped', flipped_img_with_border)
cv2.waitKey(0)
cv2.destroyAllWindows()