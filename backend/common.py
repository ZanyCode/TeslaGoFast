from pydantic import BaseModel


class Coords(BaseModel):
    current_x: int
    current_y: int
    max_x: int
    max_y: int   