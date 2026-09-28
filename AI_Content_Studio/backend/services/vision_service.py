import base64
import json
import os
from dotenv import load_dotenv
from groq import Groq

from backend.models.schemas import CreativeBrief


load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

client = Groq(api_key=groq_api_key)


def analyze_reference_image(image_bytes: bytes):
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    response = client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are Ascent AI, an expert creative director and "
                    "visual analyst. Analyze the reference image carefully "
                    "and return the visual analysis as valid JSON. "
                    "The JSON must contain information about the subject, "
                    "composition, color palette, lighting, mood, "
                    "environment, visual style, camera perspective, "
                    "and important visual details."
                )
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Analyze this reference image and return a "
                            "structured JSON visual analysis. Identify the "
                            "main subject, composition, color palette, "
                            "lighting, mood, environment, visual style, "
                            "camera perspective, and important visual "
                            "details. Return only valid JSON."
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": (
                                f"data:image/jpeg;base64,"
                                f"{image_base64}"
                            )
                        }
                    }
                ]
            }
        ],
        temperature=0.7,
        max_completion_tokens=1500,
        response_format={
            "type": "json_object"
        }
    )

    content = response.choices[0].message.content

    return json.loads(content)


def fuse_creative_brief_with_image(
    brief: CreativeBrief,
    image_bytes: bytes
):

    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    brief_data = brief.model_dump()

    response = client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are Ascent AI, an expert multimodal creative "
                    "director. You receive an existing creative brief and "
                    "a reference image. Combine them into one unified "
                    "creative brief. Preserve the original creative idea "
                    "and requirements while intelligently incorporating "
                    "useful visual inspiration from the reference image. "
                    "Do not simply describe the image. Transform its "
                    "relevant visual characteristics into creative "
                    "direction. Return only the requested JSON structure."
                )
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "EXISTING CREATIVE BRIEF:\n\n"
                            + json.dumps(brief_data)
                            + "\n\n"
                            "REFERENCE IMAGE:\n"
                            "Use the reference image as visual inspiration. "
                            "Extract relevant composition, colors, lighting, "
                            "mood, materials, environment, styling and other "
                            "useful visual characteristics.\n\n"
                            "Create a unified creative brief that combines "
                            "the original concept with the reference image."
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": (
                                f"data:image/jpeg;base64,"
                                f"{image_base64}"
                            )
                        }
                    }
                ]
            }
        ],
        temperature=0.7,
        max_completion_tokens=2000,
        reasoning_effort="low",
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "unified_creative_brief",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string"
                        },
                        "tagline": {
                            "type": "string"
                        },
                        "concept": {
                            "type": "string"
                        },
                        "mood": {
                            "type": "string"
                        },
                        "visual_style": {
                            "type": "string"
                        },
                        "target_audience": {
                            "type": "string"
                        },
                        "key_features": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            }
                        }
                    },
                    "required": [
                        "name",
                        "tagline",
                        "concept",
                        "mood",
                        "visual_style",
                        "target_audience",
                        "key_features"
                    ],
                    "additionalProperties": False
                }
            }
        }
    )

    content = response.choices[0].message.content

    data = json.loads(content)

    return CreativeBrief(**data)