#!/usr/bin/env python3
"""
Smoke test for FireRedTTS2 integration with OpenAI-compatible TTS API.
This test attempts to connect to the API (started independently) and generate speech using FireRedTTS2.
"""

import sys
import os
import time
import requests
import socket

def test_fireredtts2_api():
    """Test the OpenAI-compatible TTS API with FireRedTTS2 model."""
    api_url = "http://127.0.0.1:7778/v1/audio/speech"

    # Test data for FireRedTTS2
    test_request = {
        "model": "fireredtts2",
        "input": "Hello, this is a test of the FireRedTTS2 integration with the OpenAI-compatible API.",
        "voice": "default",
        "response_format": "wav",
        "params": {
            "temperature": 0.9,
            "topk": 30,
            "prompt_wav": None,
            "prompt_text": None,
            "model_name": "monologue",
            "device": "cuda"
        }
    }

    try:
        print("Testing FireRedTTS2 API endpoint...")

        # Make the API request
        response = requests.post(
            api_url,
            json=test_request,
            headers={"Content-Type": "application/json"},
            timeout=120  # FireRedTTS2 might take longer to load and generate
        )

        print(f"✓ API call completed with status code: {response.status_code}")

        if response.status_code == 200:
            print("✓ FireRedTTS2 API call successful!")
            print(f"✓ Response content type: {response.headers.get('content-type', 'unknown')}")
            print(f"✓ Audio data received: {len(response.content)} bytes")

            # Basic validation - check if we got audio data
            if len(response.content) > 44:  # WAV header is 44 bytes
                print("✓ Audio data appears to be valid (contains WAV header)")
                return True
            else:
                print("✗ Audio data seems too small to be valid")
                return False
        else:
            print(f"✗ API call failed with status code: {response.status_code}")
            print(f"✗ Response: {response.text}")
            return False

    except requests.exceptions.ConnectionError:
        print("✗ Could not connect to the API server. Make sure the TTS WebUI is running with the OpenAI TTS API extension enabled.")
        print("  Start the server first, then run this test.")
        return False
    except requests.exceptions.Timeout:
        print("✗ API request timed out. The FireRedTTS2 model might be taking too long to load or generate audio.")
        return False
    except Exception as e:
        print(f"✗ Unexpected error during API test: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_api_server_running():
    """Check if the API server is running on port 7778."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex(('127.0.0.1', 7778))
        sock.close()
        return result == 0
    except:
        return False


if __name__ == "__main__":
    print("FireRedTTS2 Smoke Test")
    print("=" * 50)

    # Check if server is running
    if not check_api_server_running():
        print("✗ API server is not running on http://127.0.0.1:7778")
        print("  Please start the TTS WebUI with the OpenAI TTS API extension first.")
        print("  Example: python -m tts_webui_extension.openai_tts_api")
        sys.exit(1)

    print("✓ API server is running")

    # Run the test
    success = test_fireredtts2_api()

    print("\n" + "=" * 50)
    if success:
        print("✓ Smoke test PASSED - FireRedTTS2 integration is working!")
        sys.exit(0)
    else:
        print("✗ Smoke test FAILED - Check the logs for more details")
        sys.exit(1)
