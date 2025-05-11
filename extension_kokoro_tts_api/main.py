import gradio as gr

# from .threader import (
#     activate_api,
#     deactivate_api,
#     get_api_status,
# )

PORT = 8000
HOST = "0.0.0.0"


def activate_api(host=HOST, port=PORT):
    from .api import app

    import uvicorn

    import threading

    threading.Thread(
        target=uvicorn.run,
        kwargs={"app": app, "host": host, "port": port},
        daemon=True,
    ).start()
    return {"status": "success"}


def deactivate_api():
    return {"status": "not implemented"}


def get_api_status():
    return {"status": "not implemented"}


def test_api(host=HOST, port=PORT):
    import requests

    if host == "0.0.0.0":
        host = "localhost"

    response = requests.post(
        f"http://{host}:{port}/v1/audio/speech",
        json={
            "model": "hexgrad/Kokoro-82M",
            "input": "Hello world with custom parameters.",
            "voice": "af_heart",
            "speed": 1.0,
            "params": {
                "pitch_up_key": "2",
                "index_path": "CaitArcane/added_IVF65_Flat_nprobe_1_CaitArcane_v2",
            },
        },
    )
    audio = response.content
    return audio


def test_api_with_open_ai(host=HOST, port=PORT):
    from openai import OpenAI

    if host == "0.0.0.0":
        host = "localhost"

    client = OpenAI(api_key="sk-1234567890", base_url=f"http://{host}:{port}/v1")

    with client.audio.speech.with_streaming_response.create(
        model="hexgrad/Kokoro-82M",
        voice="af_heart",
        input="Today is a wonderful day to build something people love!",
        instructions="Speak in a cheerful and positive tone.",
        extra_body={
            "params": {
                "use_gpu": True,
                # "pitch_up_key": "2",
                # "index_path": "CaitArcane/added_IVF65_Flat_nprobe_1_CaitArcane_v2",
            },
        },
    ) as response:
        import io

        audio = io.BytesIO(response.read())
        return audio.getvalue()


def ui():
    gr.Markdown("# Kokoro TTS API")
    with gr.Row():
        with gr.Column():
            gr.Markdown(
                f"""
                This extension adds an API endpoint for the Kokoro TTS model. You can use this to generate audio from text.
                
                To use the API, you need to activate it. This will start the API server and you can then use the API endpoint.
                
                The API endpoint is http://localhost:{PORT}/v1/audio/speech
                
                The OpenAI API Base_url is http://localhost:{PORT}/v1/audio/speech
                """
            )

            host = gr.Textbox(label="Host", value=HOST)
            port = gr.Number(label="Port", value=PORT)

            activate_api_btn = gr.Button("Activate API")
            activate_api_btn.click(
                fn=activate_api,
                inputs=[host, port],
                outputs=[gr.JSON()],
                api_name="activate_api",
            )

            # deactivate_api_btn = gr.Button("Deactivate API")
            # deactivate_api_btn.click(
            #     fn=deactivate_api,
            #     inputs=[],
            #     outputs=[gr.JSON()],
            #     api_name="deactivate_api",
            # )

            # get_api_status_btn = gr.Button("Get API Status")
            # get_api_status_btn.click(
            #     fn=get_api_status,
            #     inputs=[],
            #     outputs=[gr.JSON()],
            #     api_name="get_api_status",
            # )
        with gr.Column():
            test_button = gr.Button("Test API with Python requests")
            test_button.click(
                fn=test_api,
                inputs=[host, port],
                outputs=[gr.Audio()],
                api_name="test_api",
            )

            with gr.Accordion("Code Snippets", open=False):
                # show code snippet
                gr.Markdown(
                    """
            ```python
            import requests

            response = requests.post(
                "http://localhost:8000/v1/audio/speech",
                json={
                    "model": "hexgrad/Kokoro-82M",
                    "input": "Hello world with custom parameters.",
                    "voice": "af_heart",
                    "speed": 1.0,
                    "params": {
                        "pitch_up_key": "2",
                        "index_path": "CaitArcane/added_IVF65_Flat_nprobe_1_CaitArcane_v2",
                    },
                },
            )
            audio = response.content
            ```
            """
                )

            test_with_open_ai = gr.Button("Test with Python OpenAI client")
            test_with_open_ai.click(
                fn=test_api_with_open_ai,
                inputs=[host, port],
                outputs=[gr.Audio()],
                api_name="test_api_with_open_ai",
            )

            # show code snippet
            with gr.Accordion("Code Snippets", open=False):
                gr.Markdown(
                    """
            ```python
            from openai import OpenAI

            client = OpenAI(api_key="sk-1234567890", base_url="http://localhost:8000/v1")

            with client.audio.speech.with_streaming_response.create(
                model="hexgrad/Kokoro-82M",
                voice="af_heart",
                input="Today is a wonderful day to build something people love!",
                instructions="Speak in a cheerful and positive tone.",
                extra_body={
                    "params": {
                        "use_gpu": True,
                    },
                },
            ) as response:
                audio = response.read()
            ```
            """
                )


def extension__tts_generation_webui():
    """Extension entry point."""
    ui()

    return {
        "package_name": "extension_kokoro_tts_api",
        "name": "Kokoro TTS API",
        "version": "0.0.2",
        "requirements": "git+https://github.com/rsxdalv/extension_kokoro_tts_api@main",
        "description": "Kokoro TTS API is a text-to-speech model by hexgrad",
        "extension_type": "interface",
        "extension_class": "tools",
        "author": "hexgrad",
        "extension_author": "rsxdalv",
        "license": "MIT",
        "website": "https://huggingface.co/hexgrad/Kokoro-82M",
        "extension_website": "https://github.com/rsxdalv/extension_kokoro_tts_api",
        "extension_platform_version": "0.0.1",
    }


if __name__ == "__main__":
    if "demo" in locals():
        locals()["demo"].close()  # type: ignore
    with gr.Blocks() as demo:
        extension__tts_generation_webui()

    demo.launch(
        server_port=7771,
    )
