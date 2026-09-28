import json
import os
from dotenv import load_dotenv
from groq import Groq
from backend.models.schemas import CreativeBrief

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

client = Groq(api_key=groq_api_key)

def generate_creative_concept(
    content: str,
    source_type: str = "text"
) -> CreativeBrief:

    if source_type == "pdf":
        user_content = (
            "The following content was extracted from a PDF creative brief. "
            "Analyze the complete content and transform it into a professional "
            "structured creative brief.\n\n"
            f"PDF CONTENT:\n{content}"
        )
    else:
        user_content = content

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are Ascent AI, an AI creative director. "
                    "Transform the user's content into a professional "
                    "creative brief. Be imaginative, specific, and "
                    "useful for future image generation. "
                    "If the input contains detailed requirements, "
                    "preserve the important requirements while "
                    "organizing them into the requested structure."
                )
            },
            {
                "role": "user",
                "content": user_content
            }
        ],
        temperature=0.7,
        max_completion_tokens=2000,
        reasoning_effort="low",
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "creative_brief",
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