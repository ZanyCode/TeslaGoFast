import json
from operator import is_
import os
import sys
import time

from common import Coords
from detector import Detector

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

def main():
    # Python relies on the registry to guess mimetypes. Seems to be broken in some cases (see https://github.com/encode/starlette/issues/829). 
    # This is a workaround to guarantee correct mime type for js files. Necessary for hosting angular frontend app.
    import mimetypes
    mimetypes.init()
    mimetypes.add_type('application/javascript', '.js')    

    DIR_BACKEND = abspath(join(dirname(abspath(__file__))))
    DIR_STATIC_FILES = abspath(join(DIR_BACKEND, "..", "frontend", "dist", "teslagofast"))
    FILE_CONFIG = join(DIR_BACKEND, 'config.json')
  
    app = FastAPI()
    detector = Detector()


    @app.on_event("startup")
    async def run_detector():
       await detector.run()

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
        temp_frame = detector.image_full.copy()
        # camera.open(-1)
        # return_value, image = camera.read()
        cv2.rectangle(temp_frame, detector.current_speed_box[0:2], detector.current_speed_box[2:4], (0, 255, 0))
        cv2.rectangle(temp_frame, detector.speed_limit_box[0:2], detector.speed_limit_box[2:4], (0, 255, 0))
        res, im_png = cv2.imencode(".png", temp_frame)
        return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")    

    @app.get("/api/speed-image")
    async def get_speed_image():       
        res, im_png = cv2.imencode(".png", detector.image_current_speed)
        return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")    

    @app.get("/api/max-speed-image")
    async def get_speed_image():       
        res, im_png = cv2.imencode(".png", detector.image_max_speed)
        return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")    

    @app.get("/api/coords")
    async def get_coords():
       return JSONResponse((*detector.current_speed_box[0:2], *detector.speed_limit_box[0:2]))

    @app.post("/api/save")
    async def save(coords: Coords):
        save_config(FILE_CONFIG, coords)
        detector.current_speed_box = (coords.current_x, coords.current_y, coords.current_x + detector.current_speed_dims[0], coords.current_y + detector.current_speed_dims[1])
        detector.speed_limit_box = (coords.max_x, coords.max_y, coords.max_x + detector.max_speed_dims[0], coords.max_y + detector.max_speed_dims[1])

    @app.get("/api/save")
    async def save_current_image():
        # camera.open(-1)
        # return_value, image = camera.read()
        now = datetime.now() # current date and time
        filename = join(DIR_BACKEND, now.strftime("%Y-%m-%d_%H-%M-%S_snapshot.png"))
        cv2.imwrite(filename, detector.image_full)
        res, im_png = cv2.imencode(".png", detector.image_full)
        return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")    

    @app.put("/api/start-recording")
    def start_recording():
        detector.record_images = True
        return JSONResponse(detector.record_images)

    @app.put("/api/stop-recording")
    def stop_recording():
        detector.record_images = False
        return JSONResponse(detector.record_images)

    @app.get("/api/is-recording")
    def is_recording():
        return JSONResponse(detector.record_images)

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

def save_config(file, config: Coords):
    with open(file, 'w') as outfile:
        json.dump(config.dict(), outfile)    


if __name__ == "__main__":
    main()   
    