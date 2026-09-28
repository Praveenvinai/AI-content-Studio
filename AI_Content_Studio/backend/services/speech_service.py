import os

from dotenv import load_dotenv
from groq import Groq


load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

client = Groq(api_key=groq_api_key)


def transcribe_audio(
    audio_bytes: bytes,
    filename: str,
    language: str = "en"
):

    transcription = client.audio.transcriptions.create(
        file=(filename, audio_bytes),
        model="whisper-large-v3-turbo",
        language=language,
        response_format="json",
        temperature=0
    )

    return transcription.text