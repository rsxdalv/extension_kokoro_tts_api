import setuptools
from pathlib import Path

HERE = Path(__file__).parent
README = (HERE / "README.md").read_text(encoding="utf-8") if (HERE / "README.md").exists() else ""

setuptools.setup(
    name="tts_webui_extension.openai_tts_api",
    packages=setuptools.find_namespace_packages(),
    version="0.13.1",
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
            # "extension_kokoro @ git+https://github.com/rsxdalv/extension_kokoro@main",
            "extension_kokoro", # Use this line if the package is available on PyPI
        ],
        "rvc": [
            # "extension_rvc @ git+https://github.com/rsxdalv/extension_rvc@main",
            "extension_rvc", # Use this line if the package is available on PyPI
        ],
        "chatterbox": [
            # "extension_chatterbox @ git+https://github.com/rsxdalv/extension_chatterbox@main",
            "extension_chatterbox", # Use this line if the package is available on PyPI
        ],
        "kitten-tts": [
            # "extension_kitten_tts @ git+https://github.com/rsxdalv/extension_kitten_tts@main",
            "extension_kitten_tts", # Use this line if the package is available on PyPI
        ],
        "styletts2": [
            # "extension_styletts2 @ git+https://github.com/rsxdalv/extension_styletts2@main",
            "extension_styletts2", # Use this line if the package is available on PyPI
        ],
        "f5-tts": [
            # "extension_f5_tts @ git+https://github.com/rsxdalv/extension_f5_tts@main",
            "extension_f5_tts", # Use this line if the package is available on PyPI
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
