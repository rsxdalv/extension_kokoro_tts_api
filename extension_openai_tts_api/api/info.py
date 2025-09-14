"""
Information and documentation routes.
"""
import logging

from fastapi import APIRouter

logger = logging.getLogger(__name__)

router = APIRouter(tags=["info"])


@router.get("/")
async def root():
    """API information and usage examples."""
    return {
        "info": "OpenAI-compatible Text-to-Speech API with Streaming Support",
        "documentation": "/docs",
        "example": {
            "curl_regular": """
            curl -X POST http://localhost:7778/v1/audio/speech \\
                -H "Content-Type: application/json" \\
                -d '{
                    "model": "tts-1",
                    "input": "Hello world! This is a test of the text-to-speech API.",
                    "voice": "alloy"
                }' \\
                --output speech.mp3
            """,
            "curl_streaming": """
            curl -X POST http://localhost:7778/v1/audio/speech \\
                -H "Content-Type: application/json" \\
                -d '{
                    "model": "chatterbox",
                    "input": "Hello world! This is a streaming test.",
                    "voice": "random",
                    "stream": true
                }' \\
                --output speech_stream.wav
            """,
        },
    }
