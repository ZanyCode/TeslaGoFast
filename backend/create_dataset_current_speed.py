from PIL import ImageFont, ImageDraw, Image
from os.path import abspath, join, dirname
import os
import random
import numpy as np
import cv2
import time
import numpy as np
from tqdm import tqdm
# from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras.utils import Sequence

from numpy.lib.utils import info
DIR_BACKEND = abspath(join(dirname(abspath(__file__))))


def clamp(n, smallest, largest): return max(smallest, min(n, largest))
def load_blend_image():
    return np.array(Image.open(join(DIR_BACKEND, 'blend_image.png')).convert('L'))

def create_speed_text(size, font_size_range, value_range, bg_color_range, txt_color_range, min_contrast, digit_distance_range, blend_range, blend_image):
    number = random.randint(*value_range)
    bg_color = random.randint(*bg_color_range)
    txt_color = random.randint(*txt_color_range)

    if abs(bg_color - txt_color) < min_contrast:
        if (255 - bg_color) < min_contrast:
            txt_color = bg_color - min_contrast
        else:
            bg_color = txt_color + min_contrast

    # print(abs(bg_color - txt_color))
    digit_distance = random.randint(*digit_distance_range)
    font_size = random.randint(*font_size_range)
    font = ImageFont.truetype("proxima_nova.ttf", font_size)

    im = Image.new('L', size, bg_color)
    draw = ImageDraw.Draw(im)
    digits = [int(a) for a in str(number)]
    digit_sizes = [draw.textsize(str(d), font) for d in digits]
    total_width = sum([d[0] for d in digit_sizes]) + (len(digits) - 1) * digit_distance
    total_height = max([d[1] for d in digit_sizes])

    txt_position = (size[0] / 2 - total_width / 2, size[1] / 2 - total_height / 2)    
    start_x = txt_position[0]
    for digit, digit_size in zip(digits, digit_sizes):
        draw.text((start_x, txt_position[1]), str(digit), font=font, fill=txt_color)
        start_x += digit_size[0] + digit_distance    

    im_array = np.array(im)
    blend_x = random.randint(0, blend_image.shape[1] - size[0])
    blend_y = random.randint(0, blend_image.shape[0] - size[1])

    blend_im_section = blend_image[blend_y:blend_y + size[1], blend_x:blend_x + size[0]]
    blend_factor = random.uniform(*blend_range)
    im_blended = (im_array * (1 - blend_factor)) + (blend_im_section * blend_factor)

    return number, im_blended.astype(np.float32) / 255

# blend_image = load_blend_image()
# number, im = create_speed_text(
#     size=(119, 67),
#     font_size_range=(57, 57),
#     value_range=(103, 103), 
#     bg_color_range=(0, 255),
#     txt_color_range=(0, 255),
#     min_contrast = 50,      
#     digit_distance_range=(0, 5),
#     blend_range = (0.2, 0.6),
#     blend_image = blend_image,
# ) 

# def load_comparison_image():
#     im = Image.open(join(DIR_BACKEND, '2021-12-29_09-41-01_snapshot.png')).convert('L')
#     im = im.rotate(180)
#     im = im.crop((185, 255, 304, 322))
#     return im

# comparison = load_comparison_image()
# cv2.imshow('window', np.array(comparison))
# cv2.imshow('window2', im)
# cv2.waitKey()


blend_image = load_blend_image()
num_images = 14600
img_path_base = join(DIR_BACKEND, 'evaluation_data_numbers')

for i in tqdm(range(num_images)):
    number, im = create_speed_text(
        size=(119, 67),
        font_size_range=(57, 57),
        value_range=(5, 150), 
        bg_color_range=(0, 255),
        txt_color_range=(0, 255),
        min_contrast = 50,      
        digit_distance_range=(0, 5),
        blend_range = (0.2, 0.6),
        blend_image = blend_image,
    ) 
    
    formatted = (im * 255 / np.max(im)).astype('uint8')
    img = Image.fromarray(formatted)
    img_folder = join(img_path_base, str(number))
    filename = join(img_folder, str(i) + '.png')
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)
    img.save(filename)