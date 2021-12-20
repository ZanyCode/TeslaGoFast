from fastapi import FastAPI
import uvicorn
import cv2
import io
from starlette.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from datetime import datetime
from os.path import abspath, join, dirname

# Python relies on the registry to guess mimetypes. Seems to be broken in some cases (see https://github.com/encode/starlette/issues/829). 
# This is a workaround to guarantee correct mime type for js files. Necessary for hosting angular frontend app.
import mimetypes
mimetypes.init()
mimetypes.add_type('application/javascript', '.js')

DIR_BACKEND = abspath(join(dirname(abspath(__file__))))
DIR_STATIC_FILES = abspath(join(DIR_BACKEND, "..", "frontend", "dist", "teslagofast"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

camera = cv2.VideoCapture(0)

app.mount("/static", StaticFiles(directory=DIR_STATIC_FILES), name="static")

@app.get("/api/cam")
async def get_current_image():
    camera.open(-1)
    return_value, image = camera.read()
    res, im_png = cv2.imencode(".png", image)
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")    

@app.get("/api/save")
async def save_current_image():
    camera.open(-1)
    return_value, image = camera.read()
    now = datetime.now() # current date and time
    filename = join(DIR_BACKEND, now.strftime("%Y-%m-%d_%H-%M-%S_snapshot.png"))
    cv2.imwrite(filename, image)
    res, im_png = cv2.imencode(".png", image)
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")    


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    del(camera)