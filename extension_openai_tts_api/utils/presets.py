import json
import os

presets_path = os.path.join("data", "api", "presets", "global.json")


class PresetManager:
    def __init__(self):
        self.presets = self.load_presets()

    def load_presets(self):
        old_presets = {}
        if os.path.exists(presets_path):
            with open(presets_path, "r") as f:
                presets = json.load(f)
                if presets.get("_version", "0.0.0") == "1.0.0":
                    return presets
                else:
                    old_presets = presets

        presets = {
            "_version": "1.0.0",
            "global_preset": {
                "af_heart": {
                    "model": "kokoro",
                    "params": {
                        "model_name": "hexgrad/Kokoro-82M",
                        "voice": "af_heart",
                        "use_gpu": True,
                    },
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
                "chatterbox_default": {
                    "model": "chatterbox",
                    "params": {
                        "exaggeration": 0.5,
                        "cfg_weight": 0.5,
                        "temperature": 0.8,
                        "model_name": "just_a_placeholder",
                        "device": "auto",
                        "dtype": "float32",
                    },
                },
                "styletts2_default": {
                    "model": "styletts2",
                    "params": {
                        "style": "default",
                        "voice": "default",
                        "device": "auto",
                        "dtype": "float32",
                    },
                },
                "f5_tts_default": {
                    "model": "f5-tts",
                    "params": {
                        "voice": "default",
                        "device": "auto",
                    },
                },
            },
            "old_presets": old_presets,
        }
        self.set_presets(presets)
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
        presets = self.presets.get("global_preset", {})
        voice_preset = presets.get(voice_name, {})
        if voice_preset:
            return voice_preset
        else:
            raise ValueError(f"Preset {voice_name} not found, available presets: {list(presets.keys())}")

    def get_all_presets(self):
        def iter_voice_presets():
            for voice_name, voice_params in self.presets.get("global_preset", {}).items():
                yield { "value": voice_name, "label": voice_params.get("label", voice_name) }

        return list(iter_voice_presets())

preset_manager = PresetManager()
