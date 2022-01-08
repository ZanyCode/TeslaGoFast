from typing import Sequence
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
import os

# Python relies on the registry to guess mimetypes. Seems to be broken in some cases (see https://github.com/encode/starlette/issues/829). 
# This is a workaround to guarantee correct mime type for js files. Necessary for hosting angular frontend app.
import mimetypes

class Coords(BaseModel):
    current_x: int
    current_y: int
    max_x: int
    max_y: int    

def main():
    mimetypes.init()
    mimetypes.add_type('application/javascript', '.js')

    DIR_BACKEND = abspath(join(dirname(abspath(__file__))))
    DIR_STATIC_FILES = abspath(join(DIR_BACKEND, "..", "frontend", "dist", "teslagofast"))

    current_speed_dims = (128, 128)
    max_speed_dims = (128, 128)

    current_speed_box_xy = (178, 247)
    current_speed_box = (*current_speed_box_xy, current_speed_box_xy[0] + current_speed_dims[0], current_speed_box_xy[1] + current_speed_dims[1])
    max_speed_box_xy = (305, 240)
    max_speed_box = (*max_speed_box_xy, max_speed_box_xy[0] + max_speed_dims[0], max_speed_box_xy[1] + max_speed_dims[1])


    image_full = None
    image_current_speed = None
    image_max_speed = None

    record_images = False

    camera = cv2.VideoCapture(0)
    app = FastAPI()

    @app.on_event("startup")
    async def run_detector():
        # loop = asyncio.get_event_loop()
        async def get_run_task():
            nonlocal image_full, image_current_speed, image_max_speed
            base_path_current_speed_images = join(DIR_BACKEND, 'recording', 'current_speed')
            base_path_max_speed_images = join(DIR_BACKEND, 'recording', 'max_speed')
            os.makedirs(base_path_current_speed_images)
            os.makedirs(base_path_max_speed_images)
            recording_sequence_current_speed = 0
            recording_sequence_max_speed = 0
            session_id = str(uuid.uuid4())

            while(True):
                success, image = camera.read()
                # image = cv2.imread(join(DIR_BACKEND, '2021-12-29_09-41-01_snapshot.png'))
                if success:
                    image_full = cv2.rotate(image, cv2.ROTATE_180)   
                    image_current_speed = image_full[current_speed_box[1]: current_speed_box[3], current_speed_box[0]:current_speed_box[2]]
                    image_max_speed = image_full[max_speed_box[1]: max_speed_box[3], max_speed_box[0]:max_speed_box[2]]

                    if record_images:
                        im_path_current_speed = join(DIR_BACKEND, 'recording', 'current_speed', f"{session_id}_{str(recording_sequence_current_speed).zfill(6)}.png")
                        im_path_max_speed = join(DIR_BACKEND, 'recording', 'max_speed', f"{session_id}_{str(recording_sequence_max_speed).zfill(6)}.png")
                        cv2.imwrite(im_path_current_speed, image_current_speed)
                        cv2.imwrite(im_path_max_speed, image_max_speed)
                        recording_sequence_current_speed += 1
                        recording_sequence_max_speed += 1

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

    uvicorn.run(app, host="0.0.0.0", port=8000)
    del(camera)

if __name__ == "__main__":
    main()