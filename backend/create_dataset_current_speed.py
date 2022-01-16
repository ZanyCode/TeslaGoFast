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

digits_dict = {str(i): Image.open(join(DIR_BACKEND, 'data_generation', 'current_speed_templates', f'{i}.png')).convert('L') for i in range(10)}
max_text = np.array(Image.open(join(DIR_BACKEND, 'data_generation', 'current_speed_templates', 'max.png')).convert('L'))

def create_speed_text(size, value_range, bg_color_range, txt_color_range, max_color_range, min_contrast, digit_distance_range, blend_range, blend_image, blur_range, txt_shift_range, max_shift_range):
    number = random.randint(*value_range)    
    bg_color = random.randint(*bg_color_range)    
    txt_color = random.randint(*txt_color_range)
    while abs(bg_color - txt_color) < min_contrast:
        bg_color = random.randint(*bg_color_range)    
        txt_color = random.randint(*txt_color_range)

    max_color = random.randint(*max_color_range)
    blur = random.randint(*blur_range)   
    txt_shift_x = random.randint(*txt_shift_range)
    max_shift_x = random.randint(*max_shift_range)
    max_shift_y = random.randint(*max_shift_range)

    # print(abs(bg_color - txt_color))
    digit_distance = random.randint(*digit_distance_range)
    # font_size = 57
    # font = ImageFont.truetype("proxima_nova.ttf", font_size)

    im = Image.new('L', size, bg_color)
    digits = [digits_dict[a] for a in str(number)]
    digit_sizes = [d.size for d in digits]
    total_width = sum([d[0] for d in digit_sizes]) + (len(digits) - 1) * digit_distance
    # total_height = max([d[1] for d in digit_sizes])

    txt_position = (int(size[0] / 2 - total_width / 2) + txt_shift_x, 28)
    start_x = txt_position[0]
    im_array = np.array(im)
    for digit, digit_size in zip(digits, digit_sizes):
        digit_arr = np.array(digit)
        digit_pixels = digit_arr < 255
        digit_arr[digit_pixels] = txt_color
        digit_arr[~digit_pixels] = bg_color
        # im_array[start_x:start_x + digit_size[1], txt_position[1]:txt_position[1] + digit_size[0]] = np.array(digit)
        im_array[txt_position[1]:txt_position[1] + digit_size[1], start_x:start_x + digit_size[0]] = digit_arr
        # draw.text((start_x, txt_position[1]), str(digit), font=font, fill=txt_color)    
        start_x += digit_size[0] + digit_distance 

    
    max_text_pos = (85 + max_shift_x, 27 + max_shift_y)
    max_arr = np.copy(max_text)
    txt_pixels = max_arr < 255
    max_arr[txt_pixels] = max_color
    max_arr[~txt_pixels] = bg_color
    im_array[max_text_pos[0]:max_text_pos[0] + max_text.shape[0], max_text_pos[1]:max_text_pos[1] + max_text.shape[1]] = max_arr

    im_array = cv2.blur(im_array, (blur, blur))   

    # blend_x = random.randint(0, blend_image.shape[1] - size[0])
    # blend_y = random.randint(0, blend_image.shape[0] - size[1])

    # blend_im_section = blend_image[blend_y:blend_y + size[1], blend_x:blend_x + size[0]]
    # blend_factor = random.uniform(*blend_range)
    # im_blended = (im_array * (1 - blend_factor)) + (blend_im_section * blend_factor)


    # return number, im_blended.astype(np.float32) / 255
    return number, im_array.astype(np.float32) / 255

# blend_image = load_blend_image()


# def load_comparison_image():
#     im = Image.open(join(DIR_BACKEND, 'data_generation', 'current_speed_templates', '123.png')).convert('L')    
#     return im

# for i in range(100):
#     number, im = create_speed_text(
#         size=(128, 128),
#         value_range=(20, 135), 
#         bg_color_range=(0, 255),
#         txt_color_range=(0, 255),
#         max_color_range=(0, 255),
#         min_contrast = 50,      
#         digit_distance_range=(3, 11),
#         blend_range = (0.2, 0.6),
#         blend_image = blend_image,
#         blur_range=(2,3),
#         txt_shift_range=(-8, 8),
#         max_shift_range=(-5, 5)
#     ) 

#     comparison = load_comparison_image()
#     # cv2.imshow('window', np.array(comparison))
#     cv2.imshow('window2', im)
#     cv2.waitKey()


blend_image = load_blend_image()
num_images = (135-20)*1000
img_path_base = join(DIR_BACKEND, 'training', 'current_speed')

for i in tqdm(range(num_images)):
    number, im = create_speed_text(
        size=(128, 128),
        value_range=(20, 135), 
        bg_color_range=(0, 255),
        txt_color_range=(0, 255),
        max_color_range=(0, 255),
        min_contrast = 50,      
        digit_distance_range=(3, 11),
        blend_range = (0.2, 0.6),
        blend_image = blend_image,
        blur_range=(2,3),
        txt_shift_range=(-8, 8),
        max_shift_range=(-5, 5)
    ) 
    
    formatted = (im * 255 / np.max(im)).astype('uint8')
    img = Image.fromarray(formatted)
    img_folder = join(img_path_base, str(number).zfill(3))
    filename = join(img_folder, str(i) + '.png')
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)
    img.save(filename)