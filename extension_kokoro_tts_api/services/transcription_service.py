import logging
import os
import tempfile
from pathlib import Path

from fastapi import HTTPException, Request

from ..transcribe_audio_file import transcribe_audio_file


logger = logging.getLogger(__name__)


async def transcribe_audio(request: Request):
    request_id = id(request)  # Simple request ID for tracking

    try:
        logger.info(f"[{request_id}] Received transcription request")

        content_type = request.headers.get("content-type", "")
        if "multipart/form-data" not in content_type:
            raise HTTPException(
                status_code=400, detail=f"Invalid content type {content_type}"
            )

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
