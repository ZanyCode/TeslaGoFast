import cv2
import sys
import numpy as np
from os.path import abspath, join, dirname

DIR_BACKEND = abspath(join(dirname(abspath(__file__))))

def get_bounding_boxes(image):
      # Set minimum and max HSV values to display
    lower = np.array([0, 70, 0])
    upper = np.array([179, 255, 255])

    # Create HSV Image and threshold into a range.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    output = cv2.bitwise_and(image,image, mask= mask)
    x1,y1,w,h = cv2.boundingRect(~mask)
    x2 = x1+w
    y2 = y1+h
    colour = (255, 0, 0)
    thickness = 1
    cv2.rectangle(image, (x1, y1), (x2, y2), colour, thickness)

def nothing(x):
    pass

# Load in image
image = cv2.imread(join(DIR_BACKEND, 'data_small', '030', '2022-01-17_19-31-59_002299.png'))
# image = cv2.imread(join(DIR_BACKEND, 'data_small', '053', '2cb79c68-dc8e-4e72-95d2-840c5cb9c731_000467.png')) # ap on
image = cv2.imread(join(DIR_BACKEND, 'data_small', '053', '2022-01-18_09-52-55_025982.png')) # ap off, slightly blue
# image = cv2.imread(join(DIR_BACKEND, 'data_small', '053', '2022-01-17_19-31-59_012314.png')) # ap on, night mode
# image = cv2.imread(join(DIR_BACKEND, 'data_small', '053', '2022-01-17_19-31-59_015946.png')) # ap off, night mode
# image = cv2.imread(join(DIR_BACKEND, 'data_small', '053', '2022-06-25_15-26-36_014532.png')) # ap on, day mode, not really working

# Create a window
cv2.namedWindow('image')

# create trackbars for color change
cv2.createTrackbar('HMin','image',0,179,nothing) # Hue is from 0-179 for Opencv
cv2.createTrackbar('SMin','image',0,255,nothing)
cv2.createTrackbar('VMin','image',0,255,nothing)
cv2.createTrackbar('HMax','image',0,179,nothing)
cv2.createTrackbar('SMax','image',0,255,nothing)
cv2.createTrackbar('VMax','image',0,255,nothing)

# Set default value for MAX HSV trackbars.
cv2.setTrackbarPos('HMax', 'image', 179)
cv2.setTrackbarPos('SMax', 'image', 255)
cv2.setTrackbarPos('VMax', 'image', 255)

# Initialize to check if HSV min/max value changes
hMin = sMin = vMin = hMax = sMax = vMax = 0
phMin = psMin = pvMin = phMax = psMax = pvMax = 0

output = image
wait_time = 33

while(1):

    # get current positions of all trackbars
    hMin = cv2.getTrackbarPos('HMin','image')
    sMin = cv2.getTrackbarPos('SMin','image')
    vMin = cv2.getTrackbarPos('VMin','image')

    hMax = cv2.getTrackbarPos('HMax','image')
    sMax = cv2.getTrackbarPos('SMax','image')
    vMax = cv2.getTrackbarPos('VMax','image')

    # Set minimum and max HSV values to display
    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])

    # Create HSV Image and threshold into a range.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    detected_ap_pixels = np.sum(mask) / (mask.shape[0] * mask.shape[1])

    output = cv2.bitwise_and(image,image, mask= mask)

    # get_bounding_boxes(image)
    larger_mat = np.zeros((1200, 1200, 3), np.uint8)
    larger_mat[0:mask.shape[0], 0:mask.shape[1]] = output

    # larger_mat = np.zeros((1200, 1200), np.uint8)
    # larger_mat[0:mask.shape[0], 0:mask.shape[1]] = mask

    # Print if there is a change in HSV value
    if( (phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax) ):
        print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % (hMin , sMin , vMin, hMax, sMax , vMax))
        phMin = hMin
        psMin = sMin
        pvMin = vMin
        phMax = hMax
        psMax = sMax
        pvMax = vMax
        print(detected_ap_pixels)


    # Display output image
    cv2.imshow('image',larger_mat)

    # Wait longer to prevent freeze for videos.
    if cv2.waitKey(wait_time) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

