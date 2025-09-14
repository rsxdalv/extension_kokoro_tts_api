"""
Startup script for the local workspace version of the Kokoro TTS API.
This ensures we use the workspace version instead of the pip-installed package.
"""
import sys
import os

# Add the workspace to the Python path to ensure we use the local version
workspace_path = os.path.dirname(os.path.abspath(__file__))
tts_webui_path = os.path.join(workspace_path, "..", "..")
sys.path.insert(0, tts_webui_path)

def main():
    """Start the TTS API server using the local workspace version."""
    print("🚀 Starting Kokoro TTS API (Local Workspace Version)")
    print(f"📁 Workspace path: {workspace_path}")
    
    # Import and run the router
    from workspace.extension_kokoro_tts_api.extension_kokoro_tts_api.router import app
    
    # Apply torch load patch
    try:
        from tts_webui.utils.torch_load_patch import apply_torch_load_patch
        apply_torch_load_patch()
        print("✅ Torch load patch applied")
    except ImportError:
        print("⚠️  Could not apply torch load patch (optional)")
    
    # Start the server
    import uvicorn
    print("🌐 Starting server on http://0.0.0.0:7778")
    uvicorn.run(app, host="0.0.0.0", port=7778)

if __name__ == "__main__":
    main()
