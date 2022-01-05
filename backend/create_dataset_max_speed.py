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

def create_traffic_sign(size, font_size_range, values, bg_color_range, txt_color_range, ring_color_range, min_contrast, digit_distance_range, blend_range, blend_image):
    number = values[random.randint(0, len(values)-1)]
    bg_color = random.randint(*bg_color_range)
    txt_color = random.randint(*txt_color_range)
    ring_color = random.randint(*ring_color_range)

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

    distance_to_margin = 20
    ring_thickness = 8
    outer_ellipse_size = (distance_to_margin, distance_to_margin, size[0] - distance_to_margin, size[1] - distance_to_margin)
    inner_ellipse_size = (outer_ellipse_size[0] + ring_thickness, outer_ellipse_size[1] + ring_thickness, outer_ellipse_size[2] - ring_thickness, outer_ellipse_size[3] - ring_thickness)
    draw.ellipse(outer_ellipse_size, fill=ring_color)
    draw.ellipse(inner_ellipse_size, fill=bg_color)

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
# number, im = create_traffic_sign(
#     size=(128, 128),
#     font_size_range=(34, 34),
#     values=(10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 130), 
#     bg_color_range=(0, 255),
#     txt_color_range=(0, 255),
#     ring_color_range=(0, 255),
#     min_contrast = 50,      
#     digit_distance_range=(0, 0),
#     blend_range = (0.4, 0.6),
#     blend_image = blend_image,
# ) 

def load_comparison_image():
    im = Image.open(join(DIR_BACKEND, '2021-12-29_09-41-01_snapshot.png')).convert('L')
    im = im.rotate(180)
    im = im.crop((305, 240, 433, 368))
    return im

# comparison = load_comparison_image()
# cv2.imshow('window', np.array(comparison))
# cv2.imshow('window2', im)
# cv2.waitKey()


blend_image = load_blend_image()
num_images = 12000
img_path_base = join(DIR_BACKEND, 'train_data_traffic_signs')

for i in tqdm(range(num_images)):
    number, im = create_traffic_sign(
        size=(128, 128),
        font_size_range=(34, 34),
        values=(10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 130), 
        bg_color_range=(0, 255),
        txt_color_range=(0, 255),
        ring_color_range=(0, 255),
        min_contrast = 50,      
        digit_distance_range=(0, 0),
        blend_range = (0.4, 0.6),
        blend_image = blend_image,
    ) 
    
    formatted = (im * 255 / np.max(im)).astype('uint8')
    img = Image.fromarray(formatted)
    img_folder = join(img_path_base, str(number))
    filename = join(img_folder, str(i) + '.png')
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)
    img.save(filename)