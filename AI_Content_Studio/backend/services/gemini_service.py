import io
import os

from dotenv import load_dotenv
from huggingface_hub import InferenceClient


load_dotenv()

hf_token = os.getenv("HF_TOKEN")

client = InferenceClient(
    api_key=hf_token
)


def generate_image(prompt: str, aspect_ratio: str = "16:9"):
    image = client.text_to_image(
        prompt=prompt,
        model="black-forest-labs/FLUX.1-schnell"
    )
    image_buffer = io.BytesIO()
    image.save(image_buffer, format="PNG")
    return image_buffer.getvalue()