from scene_segmentation import segmentation
import cv2, ast

# read image
im = cv2.imread("./bedroom1.jpg")

# segment image
labels, masks = segmentation(im)

print(f'class id and labels: {labels}')
print(masks)
