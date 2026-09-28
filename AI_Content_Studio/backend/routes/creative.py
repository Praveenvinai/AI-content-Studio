from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import Response

from backend.models.schemas import CreativeBrief
from backend.services.groq_service import generate_creative_concept
from backend.services.image_prompt_service import generate_image_prompt
from backend.services.gemini_service import generate_image
from backend.services.pdf_service import extract_pdf_text
from backend.services.vision_service import (
    analyze_reference_image,
    fuse_creative_brief_with_image
)
from backend.services.speech_service import transcribe_audio
from backend.services.multimodal_service import (
    generate_multimodal_creative_brief
)


router = APIRouter()


@router.post("/create-concept")
def create_concept(data: dict):
    idea = data.get("idea", "").strip()
    if not idea:
        return {
            "success": False,
            "message": "Please provide a creative idea."
        }
    concept = generate_creative_concept(idea, "text")
    return {
        "success": True,
        "message": "Creative brief generated successfully!",
        "idea": idea,
        "concept": concept
    }


@router.post("/create-image-prompt")
def create_image_prompt(data: dict):
    brief_data = data.get("brief")
    if not brief_data:
        return {
            "success": False,
            "message": "Creative brief is required."
        }
    brief = CreativeBrief(**brief_data)
    image_prompt = generate_image_prompt(brief)
    return {
        "success": True,
        "message": "Image prompt generated successfully!",
        "image_prompt": image_prompt
    }


@router.post("/generate-image")
def create_image(data: dict):
    prompt = data.get("prompt", "").strip()
    aspect_ratio = data.get("aspect_ratio", "16:9")
    if not prompt:
        return {
            "success": False,
            "message": "Image prompt is required."
        }
    image_data = generate_image(
        prompt,
        aspect_ratio
    )
    return Response(
        content=image_data,
        media_type="image/png"
    )


@router.post("/upload-brief")
async def upload_brief(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        return {
            "success": False,
            "message": "Please upload a PDF file."
        }
    pdf_bytes = await file.read()
    result = extract_pdf_text(pdf_bytes)
    if not result["text"].strip():
        return {
            "success": False,
            "message": "No readable text found in the PDF."
        }
    concept = generate_creative_concept(
        result["text"],
        "pdf"
    )
    return {
        "success": True,
        "message": "PDF analyzed successfully!",
        "filename": file.filename,
        "pages": result["pages"],
        "concept": concept
    }


@router.post("/analyze-reference-image")
async def analyze_reference(file: UploadFile = File(...)):
    allowed_types = [
        "image/jpeg",
        "image/png",
        "image/webp"
    ]
    if file.content_type not in allowed_types:
        return {
            "success": False,
            "message": "Please upload a JPG, PNG, or WebP image."
        }
    image_bytes = await file.read()
    if not image_bytes:
        return {
            "success": False,
            "message": "The uploaded image is empty."
        }
    analysis = analyze_reference_image(image_bytes)
    return {
        "success": True,
        "message": "Reference image analyzed successfully!",
        "filename": file.filename,
        "analysis": analysis
    }


@router.post("/fuse-reference")
async def fuse_reference(
    brief: str = Form(...),
    file: UploadFile = File(...)
):

    allowed_types = [
        "image/jpeg",
        "image/png",
        "image/webp"
    ]

    if file.content_type not in allowed_types:
        return {
            "success": False,
            "message": "Please upload a JPG, PNG, or WebP image."
        }

    if not brief.strip():
        return {
            "success": False,
            "message": "Creative brief is required."
        }

    try:
        brief_data = CreativeBrief.model_validate_json(brief)
    except Exception:
        return {
            "success": False,
            "message": "Invalid creative brief."
        }

    image_bytes = await file.read()

    if not image_bytes:
        return {
            "success": False,
            "message": "The uploaded image is empty."
        }

    unified_brief = fuse_creative_brief_with_image(
        brief_data,
        image_bytes
    )

    return {
        "success": True,
        "message": "Creative brief successfully fused with reference image!",
        "concept": unified_brief
    }


@router.post("/transcribe-audio")
async def transcribe_audio_endpoint(
    file: UploadFile = File(...)
):
    allowed_types = [
        "audio/webm",
        "audio/wav",
        "audio/mpeg",
        "audio/mp3",
        "audio/mp4",
        "audio/x-m4a",
        "audio/ogg"
    ]
    if file.content_type not in allowed_types:
        return {
            "success": False,
            "message": "Unsupported audio format."
        }
    audio_bytes = await file.read()
    if not audio_bytes:
        return {
            "success": False,
            "message": "The recorded audio is empty."
        }
    filename = file.filename or "voice.webm"
    transcript = transcribe_audio(
        audio_bytes,
        filename
    )
    return {
        "success": True,
        "message": "Voice transcribed successfully!",
        "transcript": transcript
    }


@router.post("/multimodal-concept")
async def multimodal_concept(
    idea: str = Form(""),
    voice_transcript: str = Form(""),
    pdf_content: str = Form(""),
    reference_analysis: str = Form("")
):

    try:

        reference_data = None

        if reference_analysis.strip():
            import json
            reference_data = json.loads(
                reference_analysis
            )

        if not any([
            idea.strip(),
            voice_transcript.strip(),
            pdf_content.strip(),
            reference_data
        ]):
            return {
                "success": False,
                "message": "Please provide at least one creative input."
            }

        concept = generate_multimodal_creative_brief(
            idea=idea,
            voice_transcript=voice_transcript,
            pdf_content=pdf_content,
            reference_analysis=reference_data
        )

        return {
            "success": True,
            "message": "Multimodal creative brief generated successfully!",
            "concept": concept
        }

    except Exception as error:

        return {
            "success": False,
            "message": str(error)
        }