import logging
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import route modules
from .api import audio, models, info

logger = logging.getLogger(__name__)


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

# Include route modules
app.include_router(audio.router)
app.include_router(models.router)
app.include_router(info.router)


# OpenAI-compatible error response model
class ErrorResponse(BaseModel):
    error: dict


if __name__ == "__main__":
    from tts_webui.utils.torch_load_patch import apply_torch_load_patch

    apply_torch_load_patch()
    uvicorn.run(app, host="0.0.0.0", port=7778)
class ErrorResponse(BaseModel):
    error: dict


if __name__ == "__main__":
    from tts_webui.utils.torch_load_patch import apply_torch_load_patch

    apply_torch_load_patch()

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7778)
