"""
Text-to-speech service for handling speech generation.
"""
import logging
import tempfile
import uuid
from typing import Iterator

import ffmpeg
from fastapi import HTTPException

from ..models import CreateSpeechRequest, ResponseFormatEnum
from ..utils import to_wav_streaming_header

logger = logging.getLogger(__name__)


def generate_speech_stream(request: CreateSpeechRequest) -> Iterator[bytes]:
    """Generate speech as a stream of audio chunks"""
    if request.params:
        logger.info(f"Using custom TTS parameters: {request.params}")

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
                yield audio_chunk
            except Exception as e:
                logger.error(f"Error converting chunk: {e}")
                yield audio_chunk
    else:
        # For non-streaming models, fall back to regular generation
        result = generate_speech(request)
        yield result


def generate_speech(request: CreateSpeechRequest) -> bytes:
    """Generate speech as a single audio file (non-streaming)"""
    if request.params:
        logger.info(f"Using custom TTS parameters: {request.params}")

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
    elif model == "styletts2":
        result = styletts2_adapter(
            text,
            {
                "voice": request.voice,
                "speed": request.speed,
                **params,
            },
        )
    elif model == "f5-tts":
        result = f5_tts_adapter(
            text,
            {
                "voice": request.voice,
                "speed": request.speed,
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
    return result


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


# TTS Adapters (imported from router.py for now)
def using_with_params_decorator(func):
    def wrapper(*args, **kwargs):
        name = func.__name__
        name = name.replace("_adapter", "")
        logger.info(f"Using {name} with params: {args}, {kwargs}")
        return func(*args, **kwargs)
    return wrapper


@using_with_params_decorator
def kokoro_adapter(text, params):
    try:
        from tts_webui_extension.kokoro.main import tts
    except ImportError:
        raise ImportError(
            "Kokoro extension is not installed. Please install it to use Kokoro TTS features."
        )
    return tts(text=text, **params)


@using_with_params_decorator
def kitten_tts_adapter(text, params):
    try:
        from tts_webui_extension.kitten_tts.api import tts
    except ImportError:
        raise ImportError(
            "Kitten TTS extension is not installed. Please install it to use Kitten TTS features."
        )
    return tts(model_name="KittenML/kitten-tts-mini-0.1", text=text, **params)


@using_with_params_decorator
def chatterbox_adapter(text, params):
    try:
        from tts_webui_extension.chatterbox.api import tts
    except ImportError:
        raise ImportError(
            "Chatterbox extension is not installed. Please install it with `pip install git+https://github.com/rsxdalv/extension_chatterbox@main`"
        )
    return tts(text, **params)


@using_with_params_decorator
def styletts2_adapter(text, params):
    try:
        from tts_webui_extension.styletts2.main import tts
    except ImportError:
        raise ImportError(
            "StyleTTS2 extension is not installed. Please install it with `pip install git+https://github.com/rsxdalv/extension_styletts2@main` or your preferred source."
        )
    return tts(text, **params)


@using_with_params_decorator
def f5_tts_adapter(text, params):
    try:
        from tts_webui_extension.f5_tts.gradio_app import infer_decorated as tts
    except ImportError:
        raise ImportError(
            "F5-TTS extension is not installed. Please install it with `pip install git+https://github.com/rsxdalv/extension_f5_tts@main` or your preferred source."
        )
    return tts(text, **params)

def chatterbox_streaming_adapter(text, params) -> Iterator[bytes]:
    """Streaming adapter for chatterbox that yields audio chunks as they're generated."""
    try:
        from tts_webui_extension.chatterbox.api import tts_stream
    except ImportError:
        raise ImportError(
            "Chatterbox extension is not installed or doesn't support streaming. "
            "Please install it with `pip install git+https://github.com/rsxdalv/extension_chatterbox@main`"
        )

    logger.info(f"Using chatterbox streaming with params: {params}")

    header_sent = False
    sample_rate = None

    try:
        for partial_result in tts_stream(text, **params):
            if partial_result and "audio_out" in partial_result:
                current_sample_rate, audio_data = partial_result["audio_out"]

                if not header_sent:
                    sample_rate = current_sample_rate
                    header = to_wav_streaming_header(sample_rate)
                    yield header
                    header_sent = True

                import numpy as np

                if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
                    if np.max(np.abs(audio_data)) > 1.0:
                        audio_data = audio_data / np.max(np.abs(audio_data))
                    audio_data = (audio_data * 32767).astype(np.int16)
                elif audio_data.dtype != np.int16:
                    if np.max(np.abs(audio_data)) > 32767:
                        audio_data = audio_data * (32767 / np.max(np.abs(audio_data)))
                    audio_data = audio_data.astype(np.int16)

                yield audio_data.tobytes()

    except Exception as e:
        logger.error(f"Error in streaming chatterbox: {e}")
        result = chatterbox_adapter(text, params)
        yield webui_to_wav(result)


def generic_tts_adapter(text, params, model):
    if model == "kokoro":
        return kokoro_adapter(text, params)
    elif model == "chatterbox":
        return chatterbox_adapter(text, params)
    elif model == "kitten-tts":
        return kitten_tts_adapter(text, params)
    elif model == "styletts2":
        return styletts2_adapter(text, params)
    elif model == "f5-tts":
        return f5_tts_adapter(text, params)
    else:
        raise ValueError(f"Model {model} not found")


def preset_adapter(request: CreateSpeechRequest, text):
    from ..utils import preset_manager
    
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
        from tts_webui_extension.rvc.rvc_tab import run_rvc
    except ImportError:
        raise ImportError(
            "RVC extension is not installed. Please install it to use RVC voice conversion features."
        )

    audio = webui_to_wav(audio_result)

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
            logger.warning(f"Could not delete temporary file {temp_file.name}: {e}")


def webui_to_wav(result):
    sample_rate, audio_data = result["audio_out"]
    return to_wav(sample_rate, audio_data)


def to_wav(sample_rate, audio_data):
    from scipy.io import wavfile
    import numpy as np
    import io

    buffer = io.BytesIO()

    if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
        if np.max(np.abs(audio_data)) > 1.0:
            audio_data = audio_data / np.max(np.abs(audio_data))
        wavfile.write(buffer, sample_rate, audio_data.astype(np.float32))
    else:
        if audio_data.dtype != np.int16:
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

    format_str = format.value.lower()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_in:
        temp_in_path = temp_in.name
        temp_in.write(audio_data)
        temp_in.flush()

    temp_out_path = temp_in_path.replace(".wav", f".{format_str}")

    try:
        (
            ffmpeg.input(temp_in_path)
            .output(temp_out_path, format=format_str, strict=-2)
            .run(quiet=False, overwrite_output=True)
        )

        with open(temp_out_path, "rb") as f:
            converted_data = f.read()

        return converted_data
    except Exception as e:
        logger.error(f"Error converting audio format: {e}")
        return audio_data
    finally:
        try:
            import os
            if os.path.exists(temp_in_path):
                os.unlink(temp_in_path)
            if os.path.exists(temp_out_path):
                os.unlink(temp_out_path)
        except Exception as e:
            logger.warning(f"Could not delete temporary files: {e}")
