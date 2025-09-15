"""
Routes for managing models and voices.
"""
import logging

from fastapi import APIRouter, HTTPException

from ..services import get_voices_by_model, get_available_models

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/audio", tags=["models", "voices"])


@router.get("/models")
async def get_models():
    """Get available TTS models."""
    return {"models": get_available_models()}


@router.get("/voices/{model}")
async def get_voices(model: str):
    """Get available voices for a specific model."""
    try:
        voices = get_voices_by_model(model)
        return {"voices": voices}
    except Exception as e:
        logger.error(f"Error getting voices for model {model}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/voices")
async def get_all_voices():
    """Get all available voices from all models."""
    try:
        voices = []
        for model in ["chatterbox", "kokoro", "global_preset", "styletts2", "f5-tts"]:
            try:
                voices.extend(get_voices_by_model(model))
            except Exception as e:
                logger.warning(f"Could not get voices for model {model}: {e}")
                pass

        for voice in voices:
            voice["id"] = voice.pop("value")
            voice["name"] = voice.pop("label")

        return {"voices": voices}
    except Exception as e:
        logger.error(f"Error getting voices: {e}")
        raise HTTPException(status_code=500, detail=str(e))
