import json
import os

presets_path = os.path.join("data", "api", "presets", "global.json")


class PresetManager:
    def __init__(self):
        self.presets = self.load_presets()

    def load_presets(self):
        if os.path.exists(presets_path):
            with open(presets_path, "r") as f:
                presets = json.load(f)
        else:
            presets = {
                "global_preset": {
                    "af_heart": {
                        "model": "kokoro",
                        "model_name": "hexgrad/Kokoro-82M",
                        "voice": "af_heart",
                        "use_gpu": True,
                        # "pitch_up_key": "2",
                        # "index_path": "CaitArcane/added_IVF65_Flat_nprobe_1_CaitArcane_v2",
                    },
                },
            }
            os.makedirs(os.path.dirname(presets_path), exist_ok=True)
            with open(presets_path, "w") as f:
                json.dump(presets, f, indent=4)

        return presets

    def save_presets(self):
        os.makedirs(os.path.dirname(presets_path), exist_ok=True)
        with open(presets_path, "w") as f:
            json.dump(self.presets, f, indent=4)

    def set_preset(self, model_name, voice_name, params):
        self.presets[model_name][voice_name] = params
        self.save_presets()

    def set_presets(self, presets):
        self.presets = presets
        self.save_presets()
        return self.presets

    def get_presets(self):
        return self.presets

    def get_preset(self, model_name, voice_name):
        return self.presets.get(model_name, {}).get(voice_name, {})


preset_manager = PresetManager()
