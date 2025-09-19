import setuptools
from pathlib import Path

HERE = Path(__file__).parent
README = (HERE / "README.md").read_text(encoding="utf-8") if (HERE / "README.md").exists() else ""

setuptools.setup(
    name="tts_webui_extension.openai_tts_api",
    packages=setuptools.find_namespace_packages(),
    version="0.13.3",
    author="rsxdalv",
    description="OpenAI compatible TTS API with support for multiple TTS models",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/rsxdalv/tts_webui_extension.openai_tts_api",
    project_urls={},
    scripts=[],
    install_requires=[
        "openai",
        "gradio",
        "uvicorn",
        "psutil",
        "fastapi",
        "pydantic",
        "requests",
        "scipy",
        "numpy",
    ],
    # optional requirements
    extras_require={
        "kokoro": [
            # "tts_webui_extension.kokoro @ git+https://github.com/rsxdalv/tts_webui_extension.kokoro@main",
            "tts_webui_extension.kokoro", # Use this line if the package is available on PyPI
        ],
        "rvc": [
            # "tts_webui_extension.rvc @ git+https://github.com/rsxdalv/tts_webui_extension.rvc@main",
            "tts_webui_extension.rvc", # Use this line if the package is available on PyPI
        ],
        "chatterbox": [
            # "tts_webui_extension.chatterbox @ git+https://github.com/rsxdalv/tts_webui_extension.chatterbox@main",
            "tts_webui_extension.chatterbox", # Use this line if the package is available on PyPI
        ],
        "kitten-tts": [
            # "tts_webui_extension.kitten_tts @ git+https://github.com/rsxdalv/tts_webui_extension.kitten_tts@main",
            "tts_webui_extension.kitten_tts", # Use this line if the package is available on PyPI
        ],
        "styletts2": [
            # "tts_webui_extension.styletts2 @ git+https://github.com/rsxdalv/tts_webui_extension.styletts2@main",
            "tts_webui_extension.styletts2", # Use this line if the package is available on PyPI
        ],
        "f5-tts": [
            # "tts_webui_extension.f5_tts @ git+https://github.com/rsxdalv/tts_webui_extension.f5_tts@main",
            "tts_webui_extension.f5_tts", # Use this line if the package is available on PyPI
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
