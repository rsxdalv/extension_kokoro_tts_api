"""
Audio routes for speech generation and transcription.
"""
import logging
import uuid

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from ..models import CreateSpeechRequest
from ..services import (
    generate_speech,
    generate_speech_stream,
    get_content_type,
    transcribe_audio,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/audio", tags=["audio"])


@router.post("/speech")
async def create_speech(
    request: CreateSpeechRequest, background_tasks: BackgroundTasks
):
    """Generate speech from text using various TTS models."""
    try:
        content_type = get_content_type(request.response_format)

        if request.stream and request.model == "chatterbox":
            # Return streaming response for chatterbox
            return StreamingResponse(
                generate_speech_stream(request),
                media_type=content_type,
                headers={
                    "Content-Disposition": f"attachment; filename=speech_{uuid.uuid4()}.{request.response_format}",
                    "Transfer-Encoding": "chunked",
                },
            )
        else:
            # Return regular response for non-streaming models or when streaming is disabled
            audio_data = generate_speech(request)
            return Response(
                content=audio_data,
                media_type=content_type,
                headers={
                    "Content-Disposition": f"attachment; filename=speech_{uuid.uuid4()}.{request.response_format}"
                },
            )

    except Exception as e:
        logger.error(f"Error generating speech: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/transcriptions")
async def transcribe_audio_endpoint(request: Request):
    """Transcribe audio to text using Whisper."""
    return await transcribe_audio(request)
