from os import listdir
from os.path import abspath, join, dirname, isfile
from PIL.Image import Image
import cv2
import pytesseract
import numpy as np
import matplotlib.pyplot as plt
import time
import unittest
from typing import List, NamedTuple, Tuple
from enum import Enum

DIR_BACKEND = abspath(join(dirname(abspath(__file__))))

class DriveMode(Enum):
    Manual = 1
    Semiautomatic = 2
    Automatic = 3

class ImageInfo(NamedTuple):    
    drive_mode: DriveMode
    speed_limit: int
    drive_speed: int

def get_current_speed(img, p1, p2, drive_mode: DriveMode):
    image_section = img[p1[1]:p2[1], p1[0]:p2[0]]

    lower, upper = ((0, 25, 0), (48, 255, 82)) if drive_mode != DriveMode.Manual else ((0, 0, 0), (179, 255, 100))
    hsv = cv2.cvtColor(image_section, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)   
    speed = pytesseract.image_to_string(mask, config="nobatch digits")

    try:
        return int(speed)
    except:
        return None

def get_speed_limit(img, p1, p2):
    image_section = img[p1[1]:p2[1], p1[0]:p2[0]]

    lower, upper = (0, 0, 75), (179, 255, 255)
    hsv = cv2.cvtColor(image_section, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper) 

    cv2.imshow('wnd', mask)
    cv2.waitKey()

    speed = pytesseract.image_to_string(mask, config="nobatch digits")
    
    try:
        return int(speed)
    except:
        return None

def get_drive_mode(image, p1_steering, p2_steering, p1_current, p2_current) -> DriveMode:
    # check if steering wheel icon area contains blue pixels
    image_section = image[p1_steering[1]:p2_steering[1], p1_steering[0]:p2_steering[0]]   
    hsv = cv2.cvtColor(image_section, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (76,0,0), (120, 255, 255))
    mean_value = np.mean(mask)
    if mean_value > 10:
        return DriveMode.Automatic

    # check if selected speed area contains blue pixels
    image_section = image[p1_current[1]:p2_current[1], p1_current[0]:p2_current[0]]   
    hsv = cv2.cvtColor(image_section, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (76,0,0), (120, 255, 255))
    mean_value = np.mean(mask)

    return DriveMode.Semiautomatic if mean_value > 10 else DriveMode.Manual

def get_info_from_frame(img) -> ImageInfo:
    img = cv2.rotate(img, cv2.ROTATE_180)
    
    # Boundary points for three areas of interest (speed limit, selected speed, autopilot icon)
    p1_current, p2_current = (160, 265), (270, 370)
    shift_max = 305

    p1_max, p2_max = (p1_current[0] + shift_max, 265), (p2_current[0] + shift_max, 370)
    p1_steering, p2_steering = (316, 261), (422,366)

    drive_mode = get_drive_mode(img, p1_steering, p2_steering, p1_current, p2_current)
    speed_limit = get_speed_limit(img, p1_max, p2_max)
    speed_current = get_current_speed(img, p1_current, p2_current, drive_mode)
    return ImageInfo(drive_mode, speed_limit, speed_current)


class TestOCR(unittest.TestCase):
    def read_images(self) -> List[Tuple[str, ImageInfo]]:
        def parse_info_from_name(image_name):
            mode, speed_limit, drive_speed = image_name[20:image_name.index('.')].split('_')
            mode = DriveMode.Manual if mode == 'm' else DriveMode.Semiautomatic if mode == 's' else DriveMode.Automatic
            speed_limit = None if speed_limit == 'none' else int(speed_limit)
            drive_speed = None if drive_speed == 'none' else int(drive_speed)
            return ImageInfo(mode, speed_limit, drive_speed)

        path = join(DIR_BACKEND, 'images')
        images = [(f, parse_info_from_name(f)) for f in listdir(path) if isfile(join(path, f))]
        images = [(f, parse_info_from_name(f)) for f in listdir(path) if isfile(join(path, f))]
        return images


    def test_ocr(self):
        images = self.read_images()
        for f, expected_info in images:
            image = cv2.imread(join(DIR_BACKEND, 'images', f))
            # image = cv2.imread(join(DIR_BACKEND, 'images', '2021-12-21_10-02-58$m_50_none.png'))
            actual_info = get_info_from_frame(image)
            self.assertEqual(actual_info.drive_mode, expected_info.drive_mode, f'Drive mode incorrect for file {f}')  
            self.assertEqual(actual_info.drive_speed, expected_info.drive_speed, f'Drive speed incorrect for file {f}')  
            self.assertEqual(actual_info.speed_limit, expected_info.speed_limit, f'Speed Limit incorrect for file {f}')  

if __name__ == '__main__':
    unittest.main()




    

# img_cv = cv2.imread(join(DIR_BACKEND, 'images', '2021-12-21_10-04-10_snapshot.png'))
# img_cv = cv2.rotate( img_cv, cv2.ROTATE_180)

# img_display = img_cv.copy()

# p1_current, p2_current = (160, 265), (270, 370)
# cv2.rectangle(img_display,p1_current,p2_current,(0,255,0),1)

# shift_max = 305
# p1_max, p2_max = (p1_current[0] + shift_max, 265), (p2_current[0] + shift_max, 370)
# cv2.rectangle(img_display,p1_max,p2_max,(0,255,0),1)


# # cv2.imshow("window", img_display)
# # cv2.waitKey()

# start = time.time()
# speed_current = get_current_speed(img_cv, p1_current, p2_current)
# speed_max = get_max_speed(img_cv, p1_max, p2_max)
# end = time.time()

# print(speed_current)
# print(speed_max)
# print(end-start)
