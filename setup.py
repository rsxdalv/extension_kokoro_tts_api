import setuptools

setuptools.setup(
    name="extension_kokoro_tts_api",
    packages=setuptools.find_namespace_packages(),
    version="0.0.4",
    author="rsxdalv",
    description="Kimi Audio is a powerful text-to-speech and speech-to-text model by Moonshot AI",
    url="https://github.com/rsxdalv/extension_kokoro_tts_api",
    project_urls={},
    scripts=[],
    install_requires=[
        "extension_kokoro @ git+https://github.com/rsxdalv/extension_kokoro@main",
        "extension_rvc @ git+https://github.com/rsxdalv/extension_rvc@main",
        "openai",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
