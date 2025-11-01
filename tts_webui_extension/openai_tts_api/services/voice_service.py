"""
Voice and model management service.
"""

import logging
import os

logger = logging.getLogger(__name__)


def get_voices_by_model(model: str):
    """Get available voices for a specific model"""
    if not model:
        return []

    if model == "chatterbox":
        return get_chatterbox_voices()
    elif model == "kokoro" or model == "hexgrad/Kokoro-82M":
        return get_kokoro_voices()
    elif model == "global_preset":
        return get_global_preset_voices()
    elif model == "styletts2":
        return get_styletts2_voices()
    elif model == "f5-tts":
        return get_f5_tts_voices()
    elif model == "piper-tts":
        return get_piper_tts_voices()
    elif model == "vall-e-x":
        return get_vall_e_x_voices()
    elif model == "parler-tts":
        return get_parler_tts_voices()
    elif model == "megatts3":
        return get_megatts3_voices()
    elif model == "fireredtts2":
        return get_fireredtts2_voices()
    elif model == "higgs_v2":
        return get_higgs_v2_voices()
    elif model == "mms":
        return get_mms_voices()
    elif model == "maha_tts":
        return get_maha_tts_voices()
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
                        "value": f"voices/chatterbox/{file}",
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
        {"id": "piper-tts"},
        {"id": "vall-e-x"},
        {"id": "parler-tts"},
        {"id": "megatts3"},
        {"id": "fireredtts2"},
        {"id": "higgs_v2"},
        {"id": "mms"},
        {"id": "maha_tts"},
    ]


def get_f5_tts_voices():
    """Get available F5-TTS voices"""
    try:
        f5_dir = "voices/f5-tts"
        voices = []
        if os.path.exists(f5_dir):
            voices = [
                {
                    "value": f"{f5_dir}/{file}",
                    "label": file.replace(".wav", ""),
                }
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
                    "value": f"{styletts2_dir}/{file}",
                    "label": file.replace(".wav", ""),
                }
                for file in os.listdir(styletts2_dir)
                if file.endswith(".wav")
            ]
        return voices
    except Exception as e:
        logger.warning(f"Could not get StyleTTS2 voices: {e}")
        return []


def get_piper_tts_voices():
    """Get available Piper TTS voices"""
    # Piper TTS uses voice names, not files
    return []


def get_vall_e_x_voices():
    """Get available Vall-E-X voices"""
    return []


def get_parler_tts_voices():
    """Get available Parler TTS voices"""
    return []


def get_megatts3_voices():
    """Get available MegaTTS3 voices"""
    return []


def get_fireredtts2_voices():
    """Get available FireRedTTS2 voices"""
    return []


def get_higgs_v2_voices():
    """Get available Higgs V2 voices"""
    return []


def get_mms_voices():
    """Get available MMS voices (languages)"""
    return []


def get_maha_tts_voices():
    """Get available Maha TTS voices (speakers)"""
    try:
        from tts_webui_extension.maha_tts.api import get_voices
        return get_voices()
    except ImportError:
        logger.warning("Maha TTS extension not available")
        return []
