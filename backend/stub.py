import cv2
import numpy as np
import pytesseract
import matplotlib.pyplot as plt
from os.path import abspath, join, dirname

DIR_BACKEND = abspath(join(dirname(abspath(__file__))))
# img=cv2.imread(join(DIR_BACKEND, 'data_generation', 'current_speed_templates', '123.png'))
img=cv2.imread(join(DIR_BACKEND, 'evaluation','current_speed', '034', '2cb79c68-dc8e-4e72-95d2-840c5cb9c731_003996.png'))



# read the image and get the dimensions
h, w, _ = img.shape # assumes color image

# run tesseract, returning the bounding boxes
boxes = pytesseract.image_to_boxes(img)
print(pytesseract.image_to_string(img)) #print identified text

# draw the bounding boxes on the image
for b in boxes.splitlines():
    b = b.split()
    cv2.rectangle(img, ((int(b[1]), h - int(b[2]))), ((int(b[3]), h - int(b[4]))), (0, 255, 0), 2)

cv2.imshow('wnd', img)
cv2.waitKey()