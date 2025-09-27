"""
Voice and model management service.
"""

import logging
import os

logger = logging.getLogger(__name__)


def get_voices_by_model(model: str):
    """Get available voices for a specific model"""
    if model == "chatterbox":
        return get_chatterbox_voices()
    elif model == "kokoro":
        return get_kokoro_voices()
    elif model == "global_preset":
        return get_global_preset_voices()
    elif model == "styletts2":
        return get_styletts2_voices()
    elif model == "f5-tts":
        return get_f5_tts_voices()
    else:
        return []


def get_kokoro_voices():
    """Get available Kokoro voices"""
    try:
        from tts_webui_extension.kokoro.CHOICES import CHOICES

        voices = [{"value": key, "label": value} for key, value in CHOICES.items()]
        return voices
    except ImportError:
        logger.warning("Kokoro extension not available")
        return []


def get_chatterbox_voices():
    """Get available Chatterbox voices"""
    try:
        voices = [{"value": "random", "label": "Random"}]

        chatterbox_dir = "voices/chatterbox"
        if os.path.exists(chatterbox_dir):
            voices.extend(
                [
                    {
                        "value": os.path.join(chatterbox_dir, file),
                        "label": file.replace(".wav", ""),
                    }
                    for file in os.listdir(chatterbox_dir)
                    if file.endswith(".wav")
                ]
            )

        return voices
    except Exception as e:
        logger.warning(f"Could not get chatterbox voices: {e}")
        return [{"value": "random", "label": "Random"}]


def get_global_preset_voices():
    """Get available global preset voices"""
    try:
        from ..utils import preset_manager

        return preset_manager.get_all_presets()
    except Exception as e:
        logger.warning(f"Could not get global preset voices: {e}")
        return []


def get_available_models():
    """Get list of available TTS models"""
    return [
        {"id": "hexgrad/Kokoro-82M"},
        {"id": "chatterbox"},
        {"id": "global_preset"},
        {"id": "styletts2"},
        {"id": "f5-tts"},
    ]


def get_f5_tts_voices():
    """Get available F5-TTS voices"""
    try:
        f5_dir = "voices/f5-tts"
        voices = []
        if os.path.exists(f5_dir):
            voices = [
                {"value": os.path.join(f5_dir, file), "label": file.replace(".wav", "")}
                for file in os.listdir(f5_dir)
                if file.endswith(".wav")
            ]
        return voices
    except Exception as e:
        logger.warning(f"Could not get F5-TTS voices: {e}")
        return []


def get_styletts2_voices():
    """Get available StyleTTS2 voices"""
    try:
        styletts2_dir = "voices/styletts2"
        voices = []
        if os.path.exists(styletts2_dir):
            voices = [
                {
                    "value": os.path.join(styletts2_dir, file),
                    "label": file.replace(".wav", ""),
                }
                for file in os.listdir(styletts2_dir)
                if file.endswith(".wav")
            ]
        return voices
    except Exception as e:
        logger.warning(f"Could not get StyleTTS2 voices: {e}")
        return []
