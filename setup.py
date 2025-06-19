import setuptools

setuptools.setup(
    name="extension_kokoro_tts_api",
    packages=setuptools.find_namespace_packages(),
    version="0.8.0",
    author="rsxdalv",
    description="Kimi Audio is a powerful text-to-speech and speech-to-text model by Moonshot AI",
    url="https://github.com/rsxdalv/extension_kokoro_tts_api",
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
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
