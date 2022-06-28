import json
from pydantic import BaseModel
from os.path import exists,abspath,join,dirname

DIR_BACKEND = abspath(join(dirname(abspath(__file__))))

class Coords(BaseModel):
    current_x: int
    current_y: int
    max_x: int
    max_y: int   

class Config(BaseModel):
    current_x: int
    current_y: int
    max_x: int
    max_y: int   
    recording_path: str

def read_config(path):
    default_values = {
        "current_x": 178,
        "current_y": 247,
        "max_x": 305,
        "max_y": 240,
        "recording_path": join(DIR_BACKEND, 'recording')
    }
    
    if not exists(path):
        config = default_values
    else:
        with open(path, 'r') as f:        
            config = json.load(f)
            for k,v in default_values.items():
                if k not in config:
                    config[k] = v

    return Config(**config)

def save_config(path, config):
    with open(path, 'w') as f:
        json.dump(config.dict(), f)