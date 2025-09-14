async def transcribe_audio_file(
    file_path: str, model_name: str = "base", language: str = None
):
    from extensions.builtin.extension_whisper.main import transcribe

    # return transcribe(file_path, model_name=model_name, language=language)
    return transcribe(file_path, model_name=model_name)
