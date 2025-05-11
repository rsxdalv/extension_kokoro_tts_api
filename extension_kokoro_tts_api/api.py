from fastapi import FastAPI, HTTPException, Response, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from enum import Enum
from typing import Optional, Literal, Union, Dict, Any
import io
import uuid
import os
from datetime import datetime


# Define enums for the fixed values
# class VoiceEnum(str, Enum):
#     ALLOY = "alloy"
#     ASH = "ash"
#     BALLAD = "ballad"
#     CORAL = "coral"
#     ECHO = "echo"
#     FABLE = "fable"
#     ONYX = "onyx"
#     NOVA = "nova"
#     SAGE = "sage"
#     SHIMMER = "shimmer"
#     VERSE = "verse"

# from extension_kokoro.CHOICES import CHOICES


# class VoiceEnum(str, Enum):
#     for choice in CHOICES:
#         locals()[choice] = choice

#     class VoiceEnum(str, Enum):
#   File "c:\Users\rob\Desktop\tts-generation-webui-main\workspace\extension_kokoro_tts_api\extension_kokoro_tts_api\api.py", line 30, in VoiceEnum
#     for choice in CHOICES:
#   File "c:\Users\rob\Desktop\tts-generation-webui-main\installer_files\env\lib\enum.py", line 134, in __setitem__
#     raise TypeError('Attempted to reuse key: %r' % key)
# TypeError: Attempted to reuse key: 'choice'


class ResponseFormatEnum(str, Enum):
    MP3 = "mp3"
    OPUS = "opus"
    AAC = "aac"
    FLAC = "flac"
    WAV = "wav"
    PCM = "pcm"


class ModelEnum(str, Enum):
    TTS_1 = "tts-1"
    TTS_1_HD = "tts-1-hd"
    GPT_4O_MINI_TTS = "gpt-4o-mini-tts"


# Request model based on the specification
class CreateSpeechRequest(BaseModel):
    model: Union[ModelEnum, str] = Field(
        ..., description="One of the available TTS models"
    )
    input: str = Field(
        ..., description="The text to generate audio for", max_length=4096
    )
    # voice: VoiceEnum = Field(
    #     ..., description="The voice to use when generating the audio"
    # )
    voice: str = Field(..., description="The voice to use when generating the audio")
    response_format: ResponseFormatEnum = Field(
        default=ResponseFormatEnum.MP3, description="The format to audio in"
    )
    speed: float = Field(
        default=1.0, description="The speed of the generated audio", ge=0.25, le=4.0
    )
    instructions: Optional[str] = Field(
        None,
        description="Control the voice of your generated audio with additional instructions",
        max_length=4096,
    )
    params: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional parameters for the TTS engine"
    )

    @validator("instructions")
    def validate_instructions(cls, v, values):
        if v is not None and values.get("model") in [
            ModelEnum.TTS_1,
            ModelEnum.TTS_1_HD,
        ]:
            raise ValueError("Instructions do not work with 'tts-1' or 'tts-1-hd'")
        return v


# Create the FastAPI app
app = FastAPI(
    title="OpenAI-Compatible TTS API",
    description="A FastAPI implementation of an OpenAI-compatible Text-to-Speech endpoint",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# Placeholder for the actual TTS functionality
def generate_speech(request: CreateSpeechRequest) -> bytes:
    """
    This is a placeholder for the actual TTS implementation.
    In a real-world scenario, you would integrate with a TTS engine here.
    """
    # Mock implementation: just return a simple signal
    # In a real implementation, you would use a TTS engine like:
    # - gTTS (Google Text-to-Speech)
    # - pyttsx3
    # - Amazon Polly
    # - Microsoft Azure Cognitive Services
    # - Your own fine-tuned TTS model

    # Print out the custom params if they're provided
    if request.params:
        print(f"Using custom TTS parameters: {request.params}")
        # In a real implementation, you would pass these params to your TTS engine

    # For demo purposes, let's create a minimal audio output
    from scipy.io import wavfile
    import numpy as np

    # Create a simple sine wave as a placeholder
    sample_rate = 22050
    duration = min(2.0, len(request.input) / 20)  # Duration based on input length
    t = np.linspace(0, duration, int(sample_rate * duration), False)

    # Generate a simple tone
    freq = 440  # A4 note

    # Apply "pitch" if specified in params (just as a demo)
    if request.params and "pitch_up_key" in request.params:
        try:
            # Simple pitch shifting based on semitones
            semitones = float(request.params["pitch_up_key"])
            freq *= 2 ** (semitones / 12)
        except (ValueError, TypeError):
            pass

    signal = np.sin(freq * 2 * np.pi * t) * 0.5

    # Save to in-memory buffer
    buffer = io.BytesIO()
    wavfile.write(buffer, sample_rate, signal.astype(np.float32))
    buffer.seek(0)

    audio_data = buffer.read()

    # Convert to the requested format (in a real implementation)
    # Here we're just returning the WAV data regardless of the requested format
    return audio_data


def generate_speech_kokoro(request: CreateSpeechRequest) -> bytes:
    """
    This is a placeholder for the actual TTS implementation.
    In a real-world scenario, you would integrate with a TTS engine here.
    """
    # Mock implementation: just return a simple signal
    # In a real implementation, you would use a TTS engine like:
    # - gTTS (Google Text-to-Speech)
    # - pyttsx3
    # - Amazon Polly
    # - Microsoft Azure Cognitive Services
    # - Your own fine-tuned TTS model

    # Print out the custom params if they're provided
    if request.params:
        print(f"Using custom TTS parameters: {request.params}")
        # In a real implementation, you would pass these params to your TTS engine

    from extension_kokoro.main import tts

    params = request.params or {}

    text = request.input
    voice, speed, model_name = request.voice, request.speed, request.model

    result = tts(
        text=text,
        voice=voice,
        speed=speed,
        model_name=model_name,
        **params,
        # use_gpu=True,
    )

    sample_rate, audio_data = result["audio_out"]
    # return as wav

    from scipy.io import wavfile
    import numpy as np
    import io

    buffer = io.BytesIO()
    wavfile.write(buffer, sample_rate, audio_data.astype(np.float32))
    buffer.seek(0)

    audio_data = buffer.read()

    return audio_data


# Define the API endpoint
@app.post("/v1/audio/speech", response_class=Response)
async def create_speech(
    request: CreateSpeechRequest, background_tasks: BackgroundTasks
):
    try:
        # Generate the speech
        audio_data = generate_speech_kokoro(request)

        # Set the appropriate content type based on the requested format
        content_types = {
            ResponseFormatEnum.MP3: "audio/mpeg",
            ResponseFormatEnum.OPUS: "audio/opus",
            ResponseFormatEnum.AAC: "audio/aac",
            ResponseFormatEnum.FLAC: "audio/flac",
            ResponseFormatEnum.WAV: "audio/wav",
            ResponseFormatEnum.PCM: "audio/pcm",
        }
        content_type = content_types.get(
            request.response_format, "application/octet-stream"
        )

        # Create a response with the audio data
        return Response(
            content=audio_data,
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename=speech_{uuid.uuid4()}.{request.response_format}"
            },
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# OpenAI-compatible error response model
class ErrorResponse(BaseModel):
    error: dict


# Example usage and documentation route
@app.get("/")
async def root():
    return {
        "info": "OpenAI-compatible Text-to-Speech API",
        "documentation": "/docs",
        "example": {
            "curl": """
            curl -X POST http://localhost:8000/v1/audio/speech \\
                -H "Content-Type: application/json" \\
                -d '{
                    "model": "tts-1",
                    "input": "Hello world! This is a test of the text-to-speech API.",
                    "voice": "alloy"
                }' \\
                --output speech.mp3
            """
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7778)
