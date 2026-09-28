import json
import os

from dotenv import load_dotenv
from groq import Groq
from backend.models.schemas import CreativeBrief


load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=groq_api_key)


def generate_multimodal_creative_brief(
    idea: str = "",
    voice_transcript: str = "",
    pdf_content: str = "",
    reference_analysis: dict | None = None
) -> CreativeBrief:

    context_parts = []

    if idea.strip():
        context_parts.append(
            "TEXT IDEA:\n" + idea.strip()
        )

    if voice_transcript.strip():
        context_parts.append(
            "VOICE TRANSCRIPT:\n" + voice_transcript.strip()
        )

    if pdf_content.strip():
        context_parts.append(
            "PDF CONTENT:\n" + pdf_content.strip()
        )

    if reference_analysis:
        context_parts.append(
            "REFERENCE IMAGE ANALYSIS:\n"
            + json.dumps(reference_analysis)
        )

    combined_context = "\n\n".join(
        context_parts
    )

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are Ascent AI, an expert multimodal creative "
                    "director. You receive creative information from "
                    "multiple sources including text, voice transcripts, "
                    "PDF documents, and reference-image analysis. "
                    "Combine all relevant information into one coherent "
                    "professional creative brief. "
                    "Preserve important requirements from the user's "
                    "original idea. Use reference-image information as "
                    "visual inspiration rather than copying it literally. "
                    "Remove contradictions when possible by prioritizing "
                    "explicit user requirements. "
                    "Do not mention the internal sources or the process. "
                    "Return only the requested JSON structure."
                )
            },
            {
                "role": "user",
                "content": combined_context
            }
        ],
        temperature=0.7,
        max_completion_tokens=2000,
        reasoning_effort="low",
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "multimodal_creative_brief",
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