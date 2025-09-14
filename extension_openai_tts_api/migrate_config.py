from tts_webui.config.config_utils import get_config_value, set_config_value


def migrate_config():
    for key in ["host", "port", "auto_start"]:
        value = get_config_value("extension_kokoro_tts_api", key, None)
        if value is not None:
            set_config_value("extension_openai_tts_api", key, value)
            set_config_value("extension_kokoro_tts_api", key, None)
