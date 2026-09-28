import json
import os
from dotenv import load_dotenv
from groq import Groq
from backend.models.schemas import CreativeBrief, ImagePrompt


load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

client = Groq(api_key=groq_api_key)

def generate_image_prompt(brief: CreativeBrief) -> ImagePrompt:
    brief_data = brief.model_dump()
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert cinematic image prompt engineer. "
                    "Convert a creative brief into a highly detailed "
                    "image generation prompt. "
                    "Describe the subject, environment, composition, "
                    "lighting, atmosphere, materials, colors, camera "
                    "perspective, visual style, and important details. "
                    "Create prompts suitable for modern AI image generation."
                )
            },
            {
                "role": "user",
                "content": json.dumps(brief_data)
            }
        ],
        temperature=0.7,
        max_completion_tokens=2000,
        reasoning_effort="low",
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "image_prompt",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string"
                        },
                        "negative_prompt": {
                            "type": "string"
                        },
                        "aspect_ratio": {
                            "type": "string"
                        },
                        "style": {
                            "type": "string"
                        }
                    },
                    "required": [
                        "prompt",
                        "negative_prompt",
                        "aspect_ratio",
                        "style"
                    ],
                    "additionalProperties": False
                }
            }
        }
    )

    content = response.choices[0].message.content
    data = json.loads(content)
    return ImagePrompt(**data)