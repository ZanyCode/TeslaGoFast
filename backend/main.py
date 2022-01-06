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
    current_speed_box = (185, 255, 304, 322)
    max_speed_box = (305, 240, 433, 368)

    current_image_full = None
    current_image_speed = None
    current_image_traffic_sign = None

    camera = cv2.VideoCapture(0)
    app = FastAPI()

    @app.on_event("startup")
    async def run_detector():
        # loop = asyncio.get_event_loop()
        async def get_run_task():
            nonlocal current_image_full
            while(True):
                return_value, image = camera.read()
                # image = cv2.imread(join(DIR_BACKEND, '2021-12-29_09-41-01_snapshot.png'))
                current_image_full = cv2.rotate(image, cv2.ROTATE_180)        
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
        # camera.open(-1)
        # return_value, image = camera.read()
        cv2.rectangle(current_image_full, current_speed_box[0:2], current_speed_box[2:4], (0, 255, 0))
        cv2.rectangle(current_image_full, max_speed_box[0:2], max_speed_box[2:4], (0, 255, 0))
        res, im_png = cv2.imencode(".png", current_image_full)
        return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")    

    @app.get("/api/coords")
    async def get_coords():
       return JSONResponse((*current_speed_box[0:2], *max_speed_box[0:2]))

    @app.post("/api/save")
    async def save(coords: Coords):
        nonlocal current_speed_box, max_speed_box
        current_speed_box = (coords.current_x, coords.current_y, coords.current_x + 119, coords.current_y + 67)
        max_speed_box = (coords.max_x, coords.max_y, coords.max_x + 128, coords.max_y + 128)

    @app.get("/api/save")
    async def save_current_image():
        # camera.open(-1)
        # return_value, image = camera.read()
        now = datetime.now() # current date and time
        filename = join(DIR_BACKEND, now.strftime("%Y-%m-%d_%H-%M-%S_snapshot.png"))
        cv2.imwrite(filename, current_image_full)
        res, im_png = cv2.imencode(".png", current_image_full)
        return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")    

    uvicorn.run(app, host="0.0.0.0", port=8000)
    del(camera)

if __name__ == "__main__":
    main()