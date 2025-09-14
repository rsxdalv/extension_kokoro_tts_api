# OpenAI TTS API

This extension provides an OpenAI compatible API for multiple TTS models including Kokoro TTS and RVC.

Used as extension in [TTS Webui](https://github.com/rsxdalv/tts-webui)

## Installation

```bash
pip install git+https://github.com/rsxdalv/extension_openai_tts_api@main
```

## Requirements

Models require individual installation of their extensions.

```bash
# Kokoro TTS
pip install git+https://github.com/rsxdalv/extension_kokoro@main
# RVC
pip install git+https://github.com/rsxdalv/extension_rvc@main
# Chatterbox
pip install git+https://github.com/rsxdalv/extension_chatterbox@main
```

## Usage

### Kokoro TTS

```python
import asyncio

from openai import AsyncOpenAI
from openai.helpers import LocalAudioPlayer

openai = AsyncOpenAI()

async def main() -> None:
    async with openai.audio.speech.with_streaming_response.create(
        model="hexgrad/Kokoro-82M",
        voice="af_heart",
        input="Today is a wonderful day to build something people love!",
    ) as response:
        await LocalAudioPlayer().play(response)

if __name__ == "__main__":
    asyncio.run(main())
```

### RVC

```python
import asyncio

from openai import AsyncOpenAI
from openai.helpers import LocalAudioPlayer

openai = AsyncOpenAI()

async def main() -> None:
    async with openai.audio.speech.with_streaming_response.create(
        model="hexgrad/Kokoro-82M",
        voice="af_heart",
        input="Today is a wonderful day to build something people love!",
        extra_body={
            "params": {
                "use_gpu": True,
                "rvc_params": {
                    "pitch_up_key": "0",
                    "index_path": "CaitArcane\\added_IVF65_Flat_nprobe_1_CaitArcane_v2",
                    "pitch_collection_method": "harvest",
                    "model_path": "CaitArcane\\CaitArcane",
                    "index_rate": 0.66,
                    "filter_radius": 3,
                    "resample_sr": 0,
                    "rms_mix_rate": 1,
                    "protect": 0.33,
                },
            },
        },
    ) as response:
        await LocalAudioPlayer().play(response)

if __name__ == "__main__":
    asyncio.run(main())
```

### Chatterbox

```python
from openai import OpenAI

client = OpenAI(api_key="sk-1234567890", base_url="http://localhost:7778/v1")

with client.audio.speech.with_streaming_response.create(
    model="chatterbox",
    # voice="random",
    voice="voices/irohazaka.wav",
    input="Today is a wonderful day to build something people love!",
    # speed=1.0, # not implemented
    extra_body={
        "params": {
            "exaggeration": 0.5,
            "cfg_weight": 0.5,
            "temperature": 0.8,
        },
    },
) as response:
    audio = response.read()
    with open("audio.mp3", "wb") as f:
        f.write(audio)
```