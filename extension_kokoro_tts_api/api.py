from fastapi import FastAPI, HTTPException, Response, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from enum import Enum
from typing import Optional, Union, Dict, Any
import uuid

from .Presets import preset_manager


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


def generate_speech(request: CreateSpeechRequest) -> bytes:
    if request.params:
        print(f"Using custom TTS parameters: {request.params}")

    text = request.input
    model = request.model
    if model == "hexgrad/Kokoro-82M":
        params = request.params or {}
        result = kokoro_adapter(
            text,
            {
                "voice": request.voice,
                "speed": request.speed,
                "model_name": request.model,
                **params,
            },
        )
    elif model == "global_preset":
        result = preset_adapter(request, text)
    else:
        raise ValueError(f"Model {model} not found")

    return webui_to_wav(result)


def preset_adapter(request: CreateSpeechRequest, text):

    params_preset = preset_manager.get_preset(request.model, request.voice)

    params = {k: v for k, v in params_preset.items() if k != "model"}
    model = params_preset.get("model", None)

    if model == "kokoro":
        return kokoro_adapter(text, params)

    raise ValueError(f"Model {model} not found")


def kokoro_adapter(text, params):
    from extension_kokoro.main import tts

    print(f"Using kokoro with params: {params}")

    return tts(
        text=text,
        **params,
        # use_gpu=True,
    )


def webui_to_wav(result):
    sample_rate, audio_data = result["audio_out"]
    return to_wav(sample_rate, audio_data)


def to_wav(sample_rate, audio_data):
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
        audio_data = generate_speech(request)

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
        print(f"Error generating speech: {e}")
        # print a full stack trace
        import traceback

        traceback.print_exc()
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
