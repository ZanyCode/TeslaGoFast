import cv2
import sys
import numpy as np
from os.path import abspath, join, dirname

DIR_BACKEND = abspath(join(dirname(abspath(__file__))))


def nothing(x):
    pass

# Load in image
image = cv2.imread(join(DIR_BACKEND, 'data_generation', 'current_speed_templates', '89.png'))
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# image = cv2.rotate(image, cv2.ROTATE_180)
y, x = image.shape

# Create a window
cv2.namedWindow('image')

# create trackbars for color change
cv2.createTrackbar('XMin','image',0,x,nothing) # Hue is from 0-179 for Opencv
cv2.createTrackbar('YMin','image',0,y,nothing)
cv2.createTrackbar('IMin','image',0,255,nothing)

cv2.createTrackbar('XMax','image',0,x,nothing)
cv2.createTrackbar('YNax','image',0,y,nothing)
cv2.createTrackbar('IMax','image',0,255,nothing)

# Set default value for MAX HSV trackbars.
cv2.setTrackbarPos('XMax', 'image', x)
cv2.setTrackbarPos('YNax', 'image', y)
cv2.setTrackbarPos('IMax', 'image', 255)

# Initialize to check if HSV min/max value changes
hMin = sMin = vMin = hMax = sMax = vMax = 0
phMin = psMin = pvMin = phMax = psMax = pvMax = 0

output = image
wait_time = 33

while(1):

    # get current positions of all trackbars
    xMin = cv2.getTrackbarPos('XMin','image')
    yMin = cv2.getTrackbarPos('YMin','image')
    iMin = cv2.getTrackbarPos('IMin','image')

    xMax = cv2.getTrackbarPos('XMax','image')
    yMax = cv2.getTrackbarPos('YNax','image')
    iMax = cv2.getTrackbarPos('IMax','image')

    p1, p2 = (xMin, yMin), (xMax, yMax)

    image_section = image[p1[1]:p2[1], p1[0]:p2[0]]
    thr, image_section = cv2.threshold(image_section, iMin, iMax, cv2.THRESH_BINARY)
    larger_mat = np.zeros((500, 500), np.uint8)
    larger_mat[0:image_section.shape[0], 0:image_section.shape[1]] = image_section
    # Mat largerImage(Size(1000,1000),myImage.type());
    # largerImage = Scalar(0);
    # myImage.copyTo(largerImage(Rect(0,0,myImage.cols,myImage.rows)));    

    # Print if there is a change in HSV value
    if( (phMin != xMin) | (psMin != yMin) | (pvMin != iMin) | (phMax != xMax) | (psMax != yMax) | (pvMax != iMax) ):
        # print("(xMin = %d , yMin = %d, iMin = %d), (xMax = %d , yMax = %d, iMax = %d)" % (xMin , yMin , iMin, xMax, yMax , iMax))
        print("(%d, %d, %d, %d)" % (xMin , yMin, xMax, yMax))
        phMin = xMin
        psMin = yMin
        pvMin = iMin
        phMax = xMax
        psMax = yMax
        pvMax = iMax

    # Display output image
    cv2.imshow('image',larger_mat)
    # cv2.imshow('image', output)

    # Wait longer to prevent freeze for videos.
    if cv2.waitKey(wait_time) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()