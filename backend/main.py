import json
from operator import is_
import os
import sys
import time

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


class Coords(BaseModel):
    current_x: int
    current_y: int
    max_x: int
    max_y: int    

def main():
    # Python relies on the registry to guess mimetypes. Seems to be broken in some cases (see https://github.com/encode/starlette/issues/829). 
    # This is a workaround to guarantee correct mime type for js files. Necessary for hosting angular frontend app.
    import mimetypes
    mimetypes.init()
    mimetypes.add_type('application/javascript', '.js')

    is_linux = sys.platform.startswith("linux")
    if is_linux:
        import drivers
        display = drivers.Lcd()

    def write_lcd(txt, line):
        if is_linux:
            display.lcd_display_string(txt, line)            

    DIR_BACKEND = abspath(join(dirname(abspath(__file__))))
    DIR_STATIC_FILES = abspath(join(DIR_BACKEND, "..", "frontend", "dist", "teslagofast"))
    FILE_CONFIG = join(DIR_BACKEND, 'config.json')
    config = load_config(FILE_CONFIG)

    current_speed_dims = (128, 128)
    max_speed_dims = (128, 128)

    current_speed_box_xy = (config.current_x, config.current_y)
    current_speed_box = (*current_speed_box_xy, current_speed_box_xy[0] + current_speed_dims[0], current_speed_box_xy[1] + current_speed_dims[1])
    max_speed_box_xy = (config.max_x, config.max_y)
    max_speed_box = (*max_speed_box_xy, max_speed_box_xy[0] + max_speed_dims[0], max_speed_box_xy[1] + max_speed_dims[1])

    image_full = None
    image_current_speed = None
    image_max_speed = None

    record_images = False

    camera = cv2.VideoCapture(0)
    app = FastAPI()

    prev_current_speed = 0
    prev_max_speed = 0

    @app.on_event("startup")
    async def run_detector():
        library = 'libedgetpu.so.1' if is_linux else 'edgetpu.dll'

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
            nonlocal image_full, image_current_speed, image_max_speed, prev_max_speed, prev_current_speed
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
                success, image = camera.read()
                # image = cv2.imread(join(DIR_BACKEND, '2021-12-29_09-41-01_snapshot.png'))
                if success:
                    image_full = cv2.rotate(image, cv2.ROTATE_180)   
                    image_current_speed = image_full[current_speed_box[1]: current_speed_box[3], current_speed_box[0]:current_speed_box[2]]
                    image_max_speed = image_full[max_speed_box[1]: max_speed_box[3], max_speed_box[0]:max_speed_box[2]]                    

                    if record_images:
                        im_path_current_speed = join(DIR_BACKEND, 'recording', session_id, 'current_speed', f"{session_id}_{str(recording_sequence_idx).zfill(6)}.png")
                        cv2.imwrite(im_path_current_speed, image_current_speed)
                        recording_sequence_idx += 1
                        im_path_max_speed = join(DIR_BACKEND, 'recording', session_id, 'max_speed', f"{session_id}_{str(recording_sequence_idx).zfill(6)}.png")
                        cv2.imwrite(im_path_max_speed, image_max_speed)
                        recording_sequence_idx += 1

                    current_speed = estimate_speed(interpreter, image_current_speed)
                    max_speed = estimate_speed(interpreter, image_max_speed)

                    # Write to display
                    if prev_current_speed != current_speed or prev_max_speed != max_speed:
                        prev_current_speed = current_speed
                        write_lcd(f"{current_speed}km/h, {max_speed}km/h", 1)   
                    
                    # Print fps count
                    current_time = time.time()
                    if (current_time - last_fps_update) > 1:
                        last_fps_update = current_time
                        write_lcd(f"{frame_count} FpS", 2)
                        print(f"{frame_count} FpS")

                    frame_count += 1

                await asyncio.sleep(0.05)

        asyncio.create_task(get_run_task())


    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


    app.mount("/static", StaticFiles(directory=DIR_STATIC_FILES), name="static")

    @app.get("/api/cam")
    async def get_current_image():
        temp_frame = image_full.copy()
        # camera.open(-1)
        # return_value, image = camera.read()
        cv2.rectangle(temp_frame, current_speed_box[0:2], current_speed_box[2:4], (0, 255, 0))
        cv2.rectangle(temp_frame, max_speed_box[0:2], max_speed_box[2:4], (0, 255, 0))
        res, im_png = cv2.imencode(".png", temp_frame)
        return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")    

    @app.get("/api/speed-image")
    async def get_speed_image():       
        res, im_png = cv2.imencode(".png", image_current_speed)
        return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")    

    @app.get("/api/max-speed-image")
    async def get_speed_image():       
        res, im_png = cv2.imencode(".png", image_max_speed)
        return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")    

    @app.get("/api/coords")
    async def get_coords():
       return JSONResponse((*current_speed_box[0:2], *max_speed_box[0:2]))

    @app.post("/api/save")
    async def save(coords: Coords):
        save_config(FILE_CONFIG, coords)
        nonlocal current_speed_box, max_speed_box
        current_speed_box = (coords.current_x, coords.current_y, coords.current_x + current_speed_dims[0], coords.current_y + current_speed_dims[1])
        max_speed_box = (coords.max_x, coords.max_y, coords.max_x + max_speed_dims[0], coords.max_y + max_speed_dims[1])

    @app.get("/api/save")
    async def save_current_image():
        # camera.open(-1)
        # return_value, image = camera.read()
        now = datetime.now() # current date and time
        filename = join(DIR_BACKEND, now.strftime("%Y-%m-%d_%H-%M-%S_snapshot.png"))
        cv2.imwrite(filename, image_full)
        res, im_png = cv2.imencode(".png", image_full)
        return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")    

    @app.put("/api/start-recording")
    def start_recording():
        nonlocal record_images
        record_images = True
        return JSONResponse(record_images)

    @app.put("/api/stop-recording")
    def stop_recording():
        nonlocal record_images
        record_images = False
        return JSONResponse(record_images)

    @app.get("/api/is-recording")
    def is_recording():
        nonlocal record_images
        return JSONResponse(record_images)

    @app.post("/api/git_update")
    def update():
        repo_base_dir = join(DIR_BACKEND, '..')
        stream = os.popen(f'cd {repo_base_dir} && git fetch && git pull')
        print(stream.read())

    @app.post("/api/build_frontend")
    def build_frontend():
        frontend_dir = join(DIR_BACKEND, '..', 'frontend')
        stream = os.popen(f'cd {frontend_dir} && npx ng b')
        print(stream.read())

    @app.post("/api/reboot")
    def reboot():
        stream = os.popen('sudo service teslagofast restart')
        print(stream.read())


    uvicorn.run(app, host="0.0.0.0", port=8000)
    del(camera)


def estimate_speed(interpreter, bgr_image):      
    # Quantized Tflite model
    def set_input_tensor(interpreter, img):      
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

    def classify_image(interpreter, input):
        set_input_tensor(interpreter, input)
        interpreter.invoke()
        output_details = interpreter.get_output_details()[0]
        output = interpreter.get_tensor(output_details['index'])
        # Outputs from the TFLite model are uint8, so we dequantize the results:
        scale, zero_point = output_details['quantization']
        output = scale * (output - zero_point)
        top_1 = np.argmax(output)
        return top_1

    def prep_image(bgr_image):
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        # rgb /= 127.5
        # rgb -= 1.
        # return rgb
        rgb = rgb.astype(np.float32) / 127.5
        rgb = rgb - 1.

        return rgb

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

    idx = classify_image(interpreter, prep_image(bgr_image))
    return class_indices[idx]

def load_config(file) -> Coords:
    if os.path.exists(file):
        return Coords.parse_file(file)

    c = Coords(current_x = 178, current_y = 247, max_x = 305, max_y = 240)
    return c

def save_config(file, config: Coords):
    with open(file, 'w') as outfile:
        json.dump(config.dict(), outfile)    


if __name__ == "__main__":
    main()
    # DIR_BACKEND = abspath(join(dirname(abspath(__file__))))
    # tflite_model_path = join(DIR_BACKEND, 'tgf_quant.tflite')
    # interpreter = tf.lite.Interpreter(join(DIR_BACKEND, 'tgf_quant.tflite'))
    # interpreter.allocate_tensors()

    # current_speed_dims = (128, 128)
    # max_speed_dims = (128, 128)
    # current_speed_box_xy = (178, 247)
    # current_speed_box = (*current_speed_box_xy, current_speed_box_xy[0] + current_speed_dims[0], current_speed_box_xy[1] + current_speed_dims[1])
    # max_speed_box_xy = (305, 240)
    # max_speed_box = (*max_speed_box_xy, max_speed_box_xy[0] + max_speed_dims[0], max_speed_box_xy[1] + max_speed_dims[1])

    # image = cv2.imread(join(DIR_BACKEND, '2021-12-29_09-41-01_snapshot.png'))
    # image_full = cv2.rotate(image, cv2.ROTATE_180)   
    # image_current_speed = image_full[current_speed_box[1]: current_speed_box[3], current_speed_box[0]:current_speed_box[2]]
    # image_max_speed = image_full[max_speed_box[1]: max_speed_box[3], max_speed_box[0]:max_speed_box[2]]

    # def prep_bgr_img(bgr_img):
    #     rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    #     return prep_image_to_grayscale(rgb_img)

    # current_speed = estimate_speed(interpreter, image_current_speed)
    # print(current_speed)
    # current_speed = classify_image(interpreter, image_current_speed)
    # max_speed = classify_image(interpreter, image_max_speed)
    # print(f"Current: {current_speed}, Max: {max_speed}")        
    