#!/usr/bin/env python3
"""
Smoke test for Piper TTS integration with OpenAI-compatible TTS API.
This test attempts to connect to the API (started independently) and generate speech using Piper TTS.
"""

import sys
import os
import time
import requests
import socket

def test_piper_tts_api():
    """Test the OpenAI-compatible TTS API with Piper TTS model."""
    api_url = "http://127.0.0.1:7778/v1/audio/speech"

    # Test data for Piper TTS
    test_request = {
        "model": "piper-tts",
        "input": "Hello, this is a test of the Piper TTS integration with the OpenAI-compatible API.",
        "voice": "en_US-lessac-medium",
        "response_format": "wav",
        "speed": 1.0,
        "stream": False
    }

    print("Testing Piper TTS API endpoint...")

    try:
        # Make the API request
        response = requests.post(
            api_url,
            json=test_request,
            headers={"Content-Type": "application/json"},
            timeout=30
        )

        print(f"✓ API call completed with status code: {response.status_code}")

        if response.status_code == 200:
            print("✓ Piper TTS API call successful!")
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
        print("✗ API request timed out. The Piper TTS model might be taking too long to load or generate audio.")
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
    print("Piper TTS Smoke Test")
    print("=" * 50)

    # Check if server is running
    if not check_api_server_running():
        print("✗ API server is not running on http://127.0.0.1:7778")
        print("  Please start the TTS WebUI with the OpenAI TTS API extension first.")
        print("  Example: python -m tts_webui_extension.openai_tts_api")
        sys.exit(1)

    print("✓ API server is running")

    # Run the test
    success = test_piper_tts_api()

    print("\n" + "=" * 50)
    if success:
        print("✓ Smoke test PASSED - Piper TTS integration is working!")
        sys.exit(0)
    else:
        print("✗ Smoke test FAILED - Check the logs for more details")
        sys.exit(1)
