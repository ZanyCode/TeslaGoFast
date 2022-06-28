import json
from operator import is_
import os
from random import random
import sys
import time
import random
from common import Coords

os.environ["CUDA_VISIBLE_DEVICES"]="2" # third gpu
from typing import Sequence
import numpy as np
from fastapi import FastAPI
import uvicorn
import cv2
import io
from starlette.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from datetime import datetime
from os.path import abspath, join, dirname
from pydantic import BaseModel
import asyncio
from tflite_runtime.interpreter import load_delegate, Interpreter
import xgboost as xgb

DIR_BACKEND = abspath(join(dirname(abspath(__file__))))
FILE_CONFIG = join(DIR_BACKEND, 'config.json')

def get_quadrants(arr):
    quadrants = [M for SubA in np.split(arr,2, axis = 0) for M in np.split(SubA,2, axis = 1)]
    return quadrants

def get_features_from_image(image):
    np_img = np.array(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))        
    quadrants = [*get_quadrants(np_img[:, :, 0]), *get_quadrants(np_img[:, :, 1]), *get_quadrants(np_img[:, :, 2])]
    features = [np.mean(quadrant) for quadrant in quadrants]
    return np.array(features)

def get_features_array(dir_path):
    image_names = [join(dir_path, f) for f in listdir(dir_path) if isfile(join(dir_path, f))]
    images = [cv2.imread(name) for name in tqdm(image_names)]
    features = [get_features_from_image(img) for img in tqdm(images)]
    return features


class Detector:
    def __init__(self) -> None:
        self.is_linux = sys.platform.startswith("linux")
        self.image_full, self.image_current_speed, self.image_speed_limit = \
            None, None, None
        self.frame_count = 0
        self.recording_sequence_idx = 0
        self.camera = cv2.VideoCapture(0)
        self.current_speed, self.max_speed = 0, 0
        config = self.load_config(FILE_CONFIG)

        self.current_speed_dims = (128, 128)
        self.max_speed_dims = (128, 128)
        self.record_images = True

        self.current_speed_box_xy = (config.current_x, config.current_y)
        self.current_speed_box = (*self.current_speed_box_xy, self.current_speed_box_xy[0] + self.current_speed_dims[0], self.current_speed_box_xy[1] + self.current_speed_dims[1])
        self.max_speed_box_xy = (config.max_x, config.max_y)
        self.speed_limit_box = (*self.max_speed_box_xy, self.max_speed_box_xy[0] + self.max_speed_dims[0], self.max_speed_box_xy[1] + self.max_speed_dims[1])

        if self.is_linux:
            import drivers
            self.display = drivers.Lcd()
        self.prev_display_line1 = ""
        self.prev_display_line2 = ""

        self.interpreter = self.get_interpreter()

        self.session_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")      
            
        self.recording_sequence_idx = 0
        self.frame_count = 0
        self.last_fps_update = time.time()
        self.ap_model = xgb.Booster(model_file=join(DIR_BACKEND, 'ap_model.xgb'))

    async def run(self):       
        async def get_run_task():
            while(True):
                self.image_full = self.capture_image(False)
                self.image_speed_limit = self.extract_image_section(self.image_full, self.speed_limit_box, 22)
                self.image_current_speed = self.extract_image_section(self.image_full, self.current_speed_box, 22)
                is_ap_on = self.is_ap_on(self.image_current_speed)

                if is_ap_on:
                    speed_limit = self.estimate_speed(self.image_speed_limit)
                    current_speed = self.estimate_speed(self.image_current_speed)
                    self.save_snapshot(self.image_speed_limit, speed_limit, self.image_current_speed, current_speed)                        
                    self.update_display(True, speed_limit, current_speed)
                else:
                    self.update_display(False)   
                    self.save_snapshot(self.image_speed_limit, 42, self.image_current_speed, 43)                        


                await asyncio.sleep(0.05)

        asyncio.create_task(get_run_task())

    def capture_image(self, use_dummy=False):
        success, image = self.camera.read() if not use_dummy else (True,cv2.imread(join(DIR_BACKEND, '2021-12-29_09-41-01_snapshot.png')),)
        if success:
            return cv2.rotate(image, cv2.ROTATE_180)
        
        raise 'Error recording image'

    def extract_image_section(self, image, box, random_shift=None):
        x_rnd,y_rnd = (0,0,) if not random_shift else (random.randint(-random_shift, random_shift), random.randint(-random_shift, random_shift),)
        return image[box[1] + x_rnd : box[3] + x_rnd, box[0] + y_rnd :box[2] + y_rnd]
    
    def is_ap_on(self, img):
        # # Set minimum and max HSV values to display
        # lower = np.array([90, 0, 0])
        # upper = np.array([179, 255, 255])

        # # Create HSV Image and threshold into a range.
        # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # mask = cv2.inRange(hsv, lower, upper)
        # detected_ap_pixels = np.sum(mask) / (mask.shape[0] * mask.shape[1])
        # return detected_ap_pixels > 4
        features = get_features_from_image(img)
        pred = self.ap_model.predict(xgb.DMatrix([features]))
        return pred[0] > 0.5

    def get_interpreter(self):
        library = 'libedgetpu.so.1' if self.is_linux else 'edgetpu.dll'

        try:
            model = join(DIR_BACKEND, 'tgf_quant_edgetpu.tflite')
            interpreter = Interpreter(model, experimental_delegates=[load_delegate(library)])
            interpreter.allocate_tensors()
            print('Using edgetpu')
            return interpreter
        except:
            model = join(DIR_BACKEND, 'tgf_quant.tflite')
            interpreter = Interpreter(model)
            interpreter.allocate_tensors()
            print('Using default model')
            return interpreter

    def load_config(self, file) -> Coords:
        if os.path.exists(file):
            return Coords.parse_file(file)

        c = Coords(current_x = 178, current_y = 247, max_x = 305, max_y = 240)
        return c

    def update_display(self, is_ap_on, speed_limit=None, current_speed=None):        
        line1 = f"{current_speed}km/h, {speed_limit}km/h".ljust(16) if is_ap_on else f"No AP".ljust(16)

        current_time = time.time()
        if (current_time - self.last_fps_update) > 1:            
            self.last_fps_update = current_time
            line2 = f"{self.frame_count} FpS".ljust(16)
            self.frame_count = 0
        else:
            line2 = self.prev_display_line2
        
        if line1 != self.prev_display_line1:
            self.write_lcd(line1, 1)
        if line2 != self.prev_display_line2:
            self.write_lcd(line2, 2)
        
        self.prev_display_line1 = line1
        self.prev_display_line2 = line2
        self.frame_count += 1
    
    def write_lcd(self, txt, line):
        if self.is_linux:
            self.display.lcd_display_string(txt, line)   
        else:
            print(txt)

    def clear_lcd(self):
        if self.is_linux:
            self.display.lcd_clear()  

    def save_snapshot(self, image_speed_limit, speed_limit, image_current_speed, current_speed):
        if self.record_images:
            im_path_current_speed = join(
                DIR_BACKEND, 
                'recording', 
                self.session_id, 
                'current_speed', 
                f"{str(current_speed).zfill(3)}",  
                f"{self.session_id}_{str(self.recording_sequence_idx).zfill(6)}.png")

            if not os.path.exists(os.path.dirname(im_path_current_speed)):
                os.makedirs(os.path.dirname(im_path_current_speed))

            cv2.imwrite(im_path_current_speed, image_current_speed)
            self.recording_sequence_idx += 1

            im_path_speed_limit = join(
                DIR_BACKEND,
                'recording', 
                self.session_id,
                'speed_limit',
                f"{str(speed_limit).zfill(3)}",  
                f"{self.session_id}_{str(self.recording_sequence_idx).zfill(6)}.png")

            if not os.path.exists(os.path.dirname(im_path_speed_limit)):
                os.makedirs(os.path.dirname(im_path_speed_limit))

            cv2.imwrite(im_path_speed_limit, image_speed_limit)
            self.recording_sequence_idx += 1

    def estimate_speed(self, bgr_image):      
        class_indices = {
            0: 30,
            1: 33,
            2: 50,
            3: 53,
            4: 70,
            5: 73,
            6: 100,
            7: 103
        }

        idx = self.classify_image(self.interpreter, self.prep_image(bgr_image))
        return class_indices[idx]

    def classify_image(self, interpreter, input):
        self.set_input_tensor(interpreter, input)
        interpreter.invoke()
        output_details = interpreter.get_output_details()[0]
        output = interpreter.get_tensor(output_details['index'])
        # Outputs from the TFLite model are uint8, so we dequantize the results:
        scale, zero_point = output_details['quantization']
        output = scale * (output - zero_point)
        top_1 = np.argmax(output)
        return top_1


    def prep_image(self, bgr_image):
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        # rgb /= 127.5
        # rgb -= 1.
        # return rgb
        rgb = rgb.astype(np.float32) / 127.5
        rgb = rgb - 1.
        return rgb       

    def set_input_tensor(self, interpreter, img):      
        input_details = interpreter.get_input_details()[0]
        tensor_index = input_details['index']
        input_tensor = interpreter.tensor(tensor_index)()[0]
        # Inputs for the TFLite model must be uint8, so we quantize our input data.
        # NOTE: This step is necessary only because we're receiving input data from
        # ImageDataGenerator, which rescaled all image data to float [0,1]. When using
        # bitmap inputs, they're already uint8 [0,255] so this can be replaced with:
        #   input_tensor[:, :] = input
        scale, zero_point = input_details['quantization']
        input_tensor[:, :] = np.uint8(img / scale + zero_point)
        # input_tensor[:, :] = np.uint8(input + 1.0 / 127.5)
        # input_tensor[:, :] = input 

if __name__ == '__main__': 
    detector = Detector()
    run_task = detector.run()
    loop = asyncio.get_event_loop()
    loop.create_task(run_task)
    loop.run_forever()