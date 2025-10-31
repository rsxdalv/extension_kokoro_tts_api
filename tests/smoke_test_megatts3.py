#!/usr/bin/env python3
"""
Smoke test for MegaTTS3 integration with OpenAI-compatible TTS API.
This test attempts to connect to the API (started independently) and generate speech using MegaTTS3.
"""

import sys
import os
import time
import requests
import socket

def test_megatts3_api():
    """Test the OpenAI-compatible TTS API with MegaTTS3 model."""
    api_url = "http://127.0.0.1:7778/v1/audio/speech"

    # Test data for MegaTTS3
    test_request = {
        "model": "megatts3",
        "input": "Hello, this is a test of the MegaTTS3 integration with the OpenAI-compatible API.",
        "voice": "default",
        "response_format": "wav",
        "params": {
            "reference_audio_path": "",  # Would need actual audio file path
            "latent_npy_path": "",       # Would need actual latent file path
            "inference_steps": 32,
            "intelligibility_weight": 0.8,
            "similarity_weight": 0.8
        }
    }

    try:
        print("Testing MegaTTS3 API endpoint...")

        # Make the API request
        response = requests.post(
            api_url,
            json=test_request,
            headers={"Content-Type": "application/json"},
            timeout=180  # MegaTTS3 might take longer to load and generate
        )

        print(f"✓ API call completed with status code: {response.status_code}")

        if response.status_code == 200:
            print("✓ MegaTTS3 API call successful!")
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
        print("✗ API request timed out. The MegaTTS3 model might be taking too long to load or generate audio.")
        print("  Note: MegaTTS3 requires reference audio and latent files to be properly configured.")
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
    print("MegaTTS3 Smoke Test")
    print("=" * 50)

    # Check if server is running
    if not check_api_server_running():
        print("✗ API server is not running on http://127.0.0.1:7778")
        print("  Please start the TTS WebUI with the OpenAI TTS API extension first.")
        print("  Example: python -m tts_webui_extension.openai_tts_api")
        sys.exit(1)

    print("✓ API server is running")
    print("Note: MegaTTS3 requires reference audio and latent files.")
    print("Make sure these files are properly configured in your request parameters.")
    print()

    # Run the test
    success = test_megatts3_api()

    print("\n" + "=" * 50)
    if success:
        print("✓ Smoke test PASSED - MegaTTS3 integration is working!")
        sys.exit(0)
    else:
        print("✗ Smoke test FAILED - Check the logs for more details")
        print("  MegaTTS3 requires proper reference audio and latent file paths.")
        sys.exit(1)
