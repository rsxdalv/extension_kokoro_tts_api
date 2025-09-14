import setuptools

setuptools.setup(
    name="extension_openai_tts_api",
    packages=setuptools.find_namespace_packages(),
    version="0.12.1",
    author="rsxdalv",
    description="OpenAI compatible TTS API with support for multiple TTS models",
    url="https://github.com/rsxdalv/extension_openai_tts_api",
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
            "extension_kokoro @ git+https://github.com/rsxdalv/extension_kokoro@main",
        ],
        "rvc": [
            "extension_rvc @ git+https://github.com/rsxdalv/extension_rvc@main",
        ],
        "chatterbox": [
            "extension_chatterbox @ git+https://github.com/rsxdalv/extension_chatterbox@main",
        ],
        "kitten-tts": [
            "extension_kitten_tts @ git+https://github.com/rsxdalv/extension_kitten_tts@main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
