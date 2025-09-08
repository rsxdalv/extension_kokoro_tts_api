import logging
import os
import tempfile
import uuid
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Union

import ffmpeg
import uvicorn
from fastapi import (
    BackgroundTasks,
    FastAPI,
    HTTPException,
    Request,
    Response,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, validator



from .to_wav_streaming_header import to_wav_streaming_header
from .transcribe_audio_file import transcribe_audio_file
from .Presets import preset_manager
from .ResponseFormatEnum import ResponseFormatEnum


logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.DEBUG)


class ModelEnum(str, Enum):
    KOKORO = "hexgrad/Kokoro-82M"
    CHATTERBOX = "chatterbox"
    GLOBAL_PRESET = "global_preset"


# Request model based on the specification
class CreateSpeechRequest(BaseModel):
    model: Union[ModelEnum, str] = Field(
        ..., description="One of the available TTS models"
    )
    input: str = Field(
        ..., description="The text to generate audio for", max_length=4096
    )
    voice: str = Field(..., description="The voice to use when generating the audio")
    response_format: ResponseFormatEnum = Field(
        default=ResponseFormatEnum.WAV, description="The format to audio in"
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
    stream: bool = Field(
        default=True, description="Whether to stream the audio response"
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
    allow_origins=[
        "*",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


def generate_speech_stream(request: CreateSpeechRequest) -> Iterator[bytes]:
    """Generate speech as a stream of audio chunks"""
    if request.params:
        print(f"Using custom TTS parameters: {request.params}")

    text = request.input
    model = request.model
    params = request.params or {}

    if model == "chatterbox":
        # Use streaming chatterbox adapter
        for audio_chunk in chatterbox_streaming_adapter(
            text,
            {
                "audio_prompt_path": (
                    None if request.voice == "random" else request.voice
                ),
                "chunked": True,
                **params,
            },
        ):
            try:
                # Convert each chunk to the requested format if not WAV
                # if request.response_format != ResponseFormatEnum.WAV:
                #     converted_chunk = convert_audio_format(
                #         audio_chunk, request.response_format
                #     )
                #     yield converted_chunk
                # else:
                yield audio_chunk
            except Exception as e:
                print(f"Error converting chunk: {e}")
                # Fall back to original format if conversion fails
                yield audio_chunk
    else:
        # For non-streaming models, fall back to regular generation
        result = generate_speech(request)
        yield result


def generate_speech(request: CreateSpeechRequest) -> bytes:
    """Generate speech as a single audio file (non-streaming)"""
    if request.params:
        print(f"Using custom TTS parameters: {request.params}")

    text = request.input
    model = request.model
    params = request.params or {}

    if model == "hexgrad/Kokoro-82M":
        result = kokoro_adapter(
            text,
            {
                "voice": request.voice,
                "speed": request.speed,
                "model_name": request.model,
                **params,
            },
        )
    elif model == "chatterbox":
        result = chatterbox_adapter(
            text,
            {
                "audio_prompt_path": (
                    None if request.voice == "random" else request.voice
                ),
                "chunked": True,
                **params,
            },
        )
    elif model == "kitten-tts":
        result = kitten_tts_adapter(
            text,
            {
                "voice": request.voice,
                "speed": request.speed,
                **params,
            },
        )
    elif model == "global_preset":
        result = preset_adapter(request, text)
    else:
        raise ValueError(f"Model {model} not found")

    if params.get("rvc_params"):
        result = rvc_adapter(result, params["rvc_params"])

    result = webui_to_wav(result)

    # Convert to requested format if not WAV
    # if request.response_format != ResponseFormatEnum.WAV:
    #     result = convert_audio_format(result, request.response_format)

    return result


def generic_tts_adapter(text, params, model):
    if model == "kokoro":
        return kokoro_adapter(text, params)
    elif model == "chatterbox":
        return chatterbox_adapter(text, params)
    elif model == "kitten-tts":
        return kitten_tts_adapter(text, params)
    else:
        raise ValueError(f"Model {model} not found")


def preset_adapter(request: CreateSpeechRequest, text):
    params_preset = preset_manager.get_preset(request.model, request.voice)

    params = params_preset.get("params", {})
    model = params_preset.get("model", None)
    rvc_params = params_preset.get("rvc_params", {})

    audio_result = generic_tts_adapter(text, params, model)

    if rvc_params:
        return rvc_adapter(audio_result, rvc_params)
    else:
        return audio_result


def rvc_adapter(audio_result, rvc_params):
    import tempfile
    import os

    try:
        from extension_rvc.rvc_tab import run_rvc
    except ImportError:
        raise ImportError(
            "RVC extension is not installed. Please install it to use RVC voice conversion features."
        )

    audio = webui_to_wav(audio_result)

    # Create a temporary file that doesn't auto-delete (delete=False)
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    try:
        temp_file.write(audio)
        temp_file.flush()
        temp_file.close()

        audio_file = temp_file.name
        return run_rvc(original_audio_path=audio_file, **rvc_params)
    finally:
        try:
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
        except Exception as e:
            print(f"Warning: Could not delete temporary file {temp_file.name}: {e}")


def using_with_params_decorator(func):
    def wrapper(*args, **kwargs):
        name = func.__name__
        name = name.replace("_adapter", "")
        print(f"Using {name} with params: {args}, {kwargs}")
        return func(*args, **kwargs)

    return wrapper


@using_with_params_decorator
def kokoro_adapter(text, params):
    try:
        from extension_kokoro.main import tts
    except ImportError:
        raise ImportError(
            "Kokoro extension is not installed. Please install it to use Kokoro TTS features."
        )

    return tts(text=text, **params)

@using_with_params_decorator
def kitten_tts_adapter(text, params):
    try:
        from extension_kitten_tts.api import tts
    except ImportError:
        raise ImportError(
            "Kitten TTS extension is not installed. Please install it to use Kitten TTS features."
        )

    return tts(model_name="KittenML/kitten-tts-mini-0.1", text=text, **params)

@using_with_params_decorator
def chatterbox_adapter(text, params):
    try:
        from extension_chatterbox.api import tts
    except ImportError:
        raise ImportError(
            "Chatterbox extension is not installed. Please install it with `pip install git+https://github.com/rsxdalv/extension_chatterbox@main`"
        )

    return tts(text, **params)


def chatterbox_streaming_adapter(text, params) -> Iterator[bytes]:
    """
    Streaming adapter for chatterbox that yields audio chunks as they're generated.
    First yields a WAV header, then raw audio data chunks.
    """
    try:
        from extension_chatterbox.api import tts_stream
    except ImportError:
        raise ImportError(
            "Chatterbox extension is not installed or doesn't support streaming. "
            "Please install it with `pip install git+https://github.com/rsxdalv/extension_chatterbox@main`"
        )

    print(f"Using chatterbox streaming with params: {params}")

    header_sent = False
    sample_rate = None

    try:
        for partial_result in tts_stream(text, **params):
            if partial_result and "audio_out" in partial_result:
                current_sample_rate, audio_data = partial_result["audio_out"]

                if not header_sent:
                    # Send WAV header with first chunk
                    sample_rate = current_sample_rate
                    header = to_wav_streaming_header(sample_rate)
                    yield header
                    header_sent = True

                # Convert audio data to bytes and send
                import numpy as np

                if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
                    # Convert float to int16
                    if np.max(np.abs(audio_data)) > 1.0:
                        audio_data = audio_data / np.max(np.abs(audio_data))
                    audio_data = (audio_data * 32767).astype(np.int16)
                elif audio_data.dtype != np.int16:
                    # Convert to int16
                    if np.max(np.abs(audio_data)) > 32767:
                        audio_data = audio_data * (32767 / np.max(np.abs(audio_data)))
                    audio_data = audio_data.astype(np.int16)

                yield audio_data.tobytes()

    except Exception as e:
        print(f"Error in streaming chatterbox: {e}")
        # Fallback to non-streaming if streaming fails
        result = chatterbox_adapter(text, params)
        yield webui_to_wav(result)


def webui_to_wav(result):
    sample_rate, audio_data = result["audio_out"]
    return to_wav(sample_rate, audio_data)


def to_wav(sample_rate, audio_data):
    from scipy.io import wavfile
    import numpy as np
    import io

    buffer = io.BytesIO()

    # Check the data range to determine appropriate format
    if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
        # Data is already float, make sure it's in range [-1, 1]
        if np.max(np.abs(audio_data)) > 1.0:
            audio_data = audio_data / np.max(np.abs(audio_data))
        wavfile.write(buffer, sample_rate, audio_data.astype(np.float32))
    else:
        # Convert to int16 for integer data
        if audio_data.dtype != np.int16:
            # Scale appropriately if needed
            if np.max(np.abs(audio_data)) > 32767:
                audio_data = audio_data * (32767 / np.max(np.abs(audio_data)))
            audio_data = audio_data.astype(np.int16)
        wavfile.write(buffer, sample_rate, audio_data)

    buffer.seek(0)
    return buffer.read()


def convert_audio_format(audio_data: bytes, format: ResponseFormatEnum) -> bytes:
    """Convert audio data to the specified format"""
    if format == ResponseFormatEnum.WAV:
        return audio_data

    # Convert format enum to string
    format_str = format.value.lower()

    # For other formats, use ffmpeg
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_in:
        temp_in_path = temp_in.name
        temp_in.write(audio_data)
        temp_in.flush()

    temp_out_path = temp_in_path.replace(".wav", f".{format_str}")

    try:
        # Use ffmpeg to convert the format
        (
            # ffmpeg
            # .input(temp_in_path)
            # .output(temp_out_path, format=format_str)
            # .run(quiet=False, overwrite_output=True)
            # [opus @ 0000028343F907C0] The encoder 'opus' is experimental but experimental codecs are not enabled, add '-strict -2' if you want to use it.
            ffmpeg.input(temp_in_path)
            .output(temp_out_path, format=format_str, strict=-2)
            .run(quiet=False, overwrite_output=True)
        )

        # Read the converted file
        with open(temp_out_path, "rb") as f:
            converted_data = f.read()

        return converted_data
    except Exception as e:
        print(f"Error converting audio format: {e}")
        # Fall back to original format if conversion fails
        return audio_data
    finally:
        # Clean up temporary files
        try:
            if os.path.exists(temp_in_path):
                os.unlink(temp_in_path)
            if os.path.exists(temp_out_path):
                os.unlink(temp_out_path)
        except Exception as e:
            print(f"Warning: Could not delete temporary files: {e}")


# Add this function to get the correct MIME type
def get_content_type(format: ResponseFormatEnum) -> str:
    """Get the MIME type for the specified audio format"""
    content_types = {
        ResponseFormatEnum.MP3: "audio/mpeg",
        ResponseFormatEnum.OPUS: "audio/opus",
        ResponseFormatEnum.AAC: "audio/aac",
        ResponseFormatEnum.FLAC: "audio/flac",
        ResponseFormatEnum.WAV: "audio/wav",
        ResponseFormatEnum.PCM: "audio/pcm",
    }
    return content_types.get(format, "application/octet-stream")


# Define the API endpoint with streaming support
@app.post("/v1/audio/speech")
async def create_speech(
    request: CreateSpeechRequest, background_tasks: BackgroundTasks
):
    try:
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
        print(f"Error generating speech: {e}")
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


def get_voices_by_model(model: str):
    """Get available voices for a specific model"""
    if model == "chatterbox":
        return get_chatterbox_voices()
    elif model == "kokoro":
        return get_kokoro_voices()
    elif model == "global_preset":
        return get_global_preset_voices()
    else:
        return []


def get_kokoro_voices():
    from extension_kokoro.CHOICES import CHOICES

    voices = [{"value": key, "label": value} for key, value in CHOICES.items()]
    return voices


def _get_chatterbox_voices_broken():
    from extension_chatterbox.api import get_voices

    # This has absolute paths but we need relative ones for proper API handling
    voice_files = get_voices()

    return [{"value": "random", "label": "Random"}] + [
        {"value": filepath, "label": filename.replace(".wav", "")}
        for filename, filepath in voice_files
    ]


def get_chatterbox_voices():
    import os

    return [{"value": "random", "label": "Random"}] + [
        {
            "value": os.path.join("voices/chatterbox/", file),
            "label": file.replace(".wav", ""),
        }
        for file in os.listdir("voices/chatterbox")
        if file.endswith(".wav")
    ]


def get_global_preset_voices():
    return preset_manager.get_all_presets()


# Add a route to get voices for a specific model
@app.get("/v1/audio/voices/{model}")
async def get_voices(model: str):
    try:
        voices = get_voices_by_model(model)
        return {"voices": voices}
    except Exception as e:
        print(f"Error getting voices for model {model}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/audio/voices")
async def get_all_voices():
    try:
        voices = []
        for model in ["chatterbox", "kokoro", "global_preset"]:
            try:
                voices.extend(get_voices_by_model(model))
            except Exception as e:
                pass
        for voice in voices:
            voice["id"] = voice.pop("value")
            voice["name"] = voice.pop("label")
        return {"voices": voices}
    except Exception as e:
        print(f"Error getting voices: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/audio/models")
async def get_models():
    return {
        "models": [
            {"id": "hexgrad/Kokoro-82M"},
            {"id": "chatterbox"},
            {"id": "global_preset"},
        ]
    }

@app.post("/v1/audio/transcriptions")
async def transcribe_audio(request: Request):
    request_id = id(request)  # Simple request ID for tracking

    try:
        logger.info(f"[{request_id}] Received transcription request")

        content_type = request.headers.get("content-type", "")
        if "multipart/form-data" not in content_type:
            raise HTTPException(status_code=400, detail=f"Invalid content type {content_type}")

        form = await request.form()
        logger.debug(f"[{request_id}] Form fields: {list(form.keys())}")
        audio_file = form.get("file")
        model = form.get("model", "whisper-1")

        if not audio_file:
            raise HTTPException(status_code=400, detail="Missing audio_file")

        # Handle file for Whisper transcription
        tmp_path = None
        try:
            # Get file extension for proper temp file handling
            filename = getattr(audio_file, "filename", "audio.webm")
            file_ext = Path(filename).suffix or ".webm"

            logger.debug(f"[{request_id}] File extension: {file_ext}")

            # Create temporary file with proper extension
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                # Read file contents
                if hasattr(audio_file, "read"):
                    # UploadFile object
                    contents = await audio_file.read()
                    logger.debug(
                        f"[{request_id}] Read {len(contents)} bytes from upload file"
                    )
                else:
                    # Handle other file-like objects
                    contents = audio_file
                    logger.debug(
                        f"[{request_id}] Using provided file data: {len(contents)} bytes"
                    )

                tmp.write(contents)
                tmp.flush()
                tmp_path = tmp.name

            logger.info(f"[{request_id}] Created temporary file: {tmp_path}")

            # Call the transcription service with file path
            transcription = await transcribe_audio_file(tmp_path, model)

        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

        return {"text": transcription}

    except HTTPException:
        # Re-raise HTTP exceptions without logging as errors
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Transcription failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Internal server error during transcription"
        )


if __name__ == "__main__":
    from tts_webui.utils.torch_load_patch import apply_torch_load_patch

    apply_torch_load_patch()

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7778)
