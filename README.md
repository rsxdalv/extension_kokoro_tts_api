# Kokoro TTS API

This extension provides an OpenAI compatible API for Kokoro TTS and RVC.

Used as extension in [TTS Webui](https://github.com/rsxdalv/tts-webui)

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
