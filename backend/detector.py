import json
from operator import is_
import os
import sys
import time

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
import uuid
from tflite_runtime.interpreter import load_delegate, Interpreter

DIR_BACKEND = abspath(join(dirname(abspath(__file__))))
FILE_CONFIG = join(DIR_BACKEND, 'config.json')

class Detector:
    def __init__(self) -> None:
        self.is_linux = sys.platform.startswith("linux")
        self.image_full, self.image_current_speed, self.image_max_speed, self.prev_max_speed, self.prev_current_speed = \
            None, None, None, 0, 0
        self.camera = cv2.VideoCapture(0)
        self.current_speed, self.max_speed = 0, 0
        config = self.load_config(FILE_CONFIG)

        self.current_speed_dims = (128, 128)
        self.max_speed_dims = (128, 128)
        self.record_images = False

        self.current_speed_box_xy = (config.current_x, config.current_y)
        self.current_speed_box = (*self.current_speed_box_xy, self.current_speed_box_xy[0] + self.current_speed_dims[0], self.current_speed_box_xy[1] + self.current_speed_dims[1])
        self.max_speed_box_xy = (config.max_x, config.max_y)
        self.max_speed_box = (*self.max_speed_box_xy, self.max_speed_box_xy[0] + self.max_speed_dims[0], self.max_speed_box_xy[1] + self.max_speed_dims[1])

        if self.is_linux:
            import drivers
            self.display = drivers.Lcd()

    async def run(self):
        library = 'libedgetpu.so.1' if self.is_linux else 'edgetpu.dll'

        try:
            model = join(DIR_BACKEND, 'tgf_quant_edgetpu.tflite')
            interpreter = Interpreter(model, experimental_delegates=[load_delegate(library)])
            interpreter.allocate_tensors()
            print('Using edgetpu')
        except:
            model = join(DIR_BACKEND, 'tgf_quant.tflite')
            interpreter = Interpreter(model)
            interpreter.allocate_tensors()
            print('Using default model')

        async def get_run_task():
            # session_id = str(uuid.uuid4())
            session_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            base_path_current_speed_images = join(DIR_BACKEND, 'recording', session_id, 'current_speed')
            base_path_max_speed_images = join(DIR_BACKEND, 'recording', session_id, 'max_speed')
            if not os.path.exists(base_path_current_speed_images):
                os.makedirs(base_path_current_speed_images)

            if not os.path.exists(base_path_max_speed_images):
                os.makedirs(base_path_max_speed_images)
                
            recording_sequence_idx = 0
            frame_count = 0
            last_fps_update = time.time()

            while(True):
                success, image = self.camera.read()
                image = cv2.imread(join(DIR_BACKEND, '2021-12-29_09-41-01_snapshot.png'))
                if success:
                    self.image_full = cv2.rotate(image, cv2.ROTATE_180)   
                    self.image_current_speed = self.image_full[self.current_speed_box[1]: self.current_speed_box[3], self.current_speed_box[0]:self.current_speed_box[2]]
                    self.image_max_speed = self.image_full[self.max_speed_box[1]: self.max_speed_box[3], self.max_speed_box[0]:self.max_speed_box[2]]                    

                    if self.record_images:
                        im_path_current_speed = join(DIR_BACKEND, 'recording', session_id, 'current_speed', f"{session_id}_{str(recording_sequence_idx).zfill(6)}.png")
                        cv2.imwrite(im_path_current_speed, self.image_current_speed)
                        recording_sequence_idx += 1
                        im_path_max_speed = join(DIR_BACKEND, 'recording', session_id, 'max_speed', f"{session_id}_{str(recording_sequence_idx).zfill(6)}.png")
                        cv2.imwrite(im_path_max_speed, self.image_max_speed)
                        recording_sequence_idx += 1

                    self.current_speed = self.estimate_speed(interpreter, self.image_current_speed)
                    self.max_speed = self.estimate_speed(interpreter, self.image_max_speed)

                    # Write to display
                    if self.prev_current_speed != self.current_speed or self.prev_max_speed != self.max_speed:
                        self.prev_current_speed = self.current_speed
                        self.write_lcd(f"{self.current_speed}km/h, {self.max_speed}km/h", 1)   
                    
                    # Print fps count
                    current_time = time.time()
                    if (current_time - last_fps_update) > 1:
                        last_fps_update = current_time
                        self.write_lcd(f"{frame_count} FpS", 2)
                        print(f"{frame_count} FpS")
                        frame_count = 0

                    frame_count += 1

                await asyncio.sleep(0.05)

        asyncio.create_task(get_run_task())

    def load_config(self, file) -> Coords:
        if os.path.exists(file):
            return Coords.parse_file(file)

        c = Coords(current_x = 178, current_y = 247, max_x = 305, max_y = 240)
        return c

    
    def write_lcd(self, txt, line):
        if self.is_linux:
            self.display.lcd_display_string(txt, line)   

    def estimate_speed(self, interpreter, bgr_image):      
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

        idx = self.classify_image(interpreter, self.prep_image(bgr_image))
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