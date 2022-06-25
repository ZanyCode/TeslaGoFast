import cv2
import numpy as np
import pytesseract
import matplotlib.pyplot as plt
from os.path import abspath, join, dirname
import time

# from train_number_recognizer import get_number_dataset_validation

DIR_BACKEND = abspath(join(dirname(abspath(__file__))))
# img=cv2.imread(join(DIR_BACKEND, 'data_generation', 'current_speed_templates', '123.png'))
img = cv2.imread(join(DIR_BACKEND, 'data_small', '103', '2cb79c68-dc8e-4e72-95d2-840c5cb9c731_000022.png'))
img = cv2.imread(join(DIR_BACKEND, 'data_small', '053', '2cb79c68-dc8e-4e72-95d2-840c5cb9c731_000467.png'))
# img = cv2.imread(join(DIR_BACKEND, 'data_small', '073', '2cb79c68-dc8e-4e72-95d2-840c5cb9c731_001594.png'))
# read the image and get the dimensions
h, w, _ = img.shape # assumes color image

# Set minimum and max HSV values to display
lower = np.array([0, 0, 71])
upper = np.array([179, 255, 255])

# dataset = get_number_dataset_validation(join(DIR_BACKEND, 'data_small'), batch_size=100000)
# images, labels = next(dataset)

# for img in images:
# img = np.array((img + 1 ) * 127, dtype=np.uint8)
# img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# Create HSV Image and threshold into a range.
start = time.time()

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, lower, upper)

# x,y,w,h = cv2.boundingRect(~mask)

# cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
contours = cv2.findContours(~mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]
colour = (255, 0, 0)
thickness = 1
i = 0
for cntr in contours:
    x1,y1,w,h = cv2.boundingRect(cntr)
    x2 = x1+w
    y2 = y1+h
    cv2.rectangle(img, (x1, y1), (x2, y2), colour, thickness)
    print("Object:", i+1, "x1:", x1, "x2:", x2, "y1:", y1, "y2:", y2)
    i += 1

print(time.time() - start)

# Display output image
cv2.imshow('image',mask)

# Wait longer to prevent freeze for videos.
cv2.waitKey() 
