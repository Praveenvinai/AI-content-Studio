from pydantic import BaseModel
from typing import List


class CreativeBrief(BaseModel):
    name: str
    tagline: str
    concept: str
    mood: str
    visual_style: str
    target_audience: str
    key_features: List[str]


class ImagePrompt(BaseModel):
    prompt: str
    negative_prompt: str
    aspect_ratio: str
    style: str