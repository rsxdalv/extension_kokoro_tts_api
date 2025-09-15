from multiprocessing import Process
import psutil
import time
import uvicorn

# Constants
HOST = "127.0.0.1"
PORT = 8765

# Global variable to store the server process
api_server_process = None


def activate_api(host, port):
    """
    Activate the API server using a separate process.

    Returns:
        dict: Status information about the activation attempt
    """
    global HOST, PORT
    HOST = host
    PORT = port
    global api_server_process

    # Check if the API is already running
    if api_server_process is not None and api_server_process.is_alive():
        return {
            "status": "already_running",
            "message": f"API is already active on {HOST}:{PORT}",
        }

    # Start the server in a separate process
    api_server_process = Process(
        target=uvicorn.run,
        kwargs={
            "app": "workspace.extension_openai_tts_api.extension_openai_tts_api.api:app",
            "host": HOST,
            "port": PORT,
        },
    )
    api_server_process.daemon = (
        True  # Make process a daemon so it exits when main program does
    )
    api_server_process.start()

    # Give some time for the server to initialize
    time.sleep(2)

    # Check if the process is running
    if api_server_process.is_alive():
        return {
            "status": "success",
            "message": f"API activated on {HOST}:{PORT} with PID {api_server_process.pid}",
        }
    else:
        return {"status": "error", "message": "Failed to start API server"}


def deactivate_api():
    """
    Deactivate the running API server process and all its child processes.

    Returns:
        dict: Status information about the deactivation attempt
    """
    global api_server_process

    # Check if the API is running
    if api_server_process is None or not api_server_process.is_alive():
        return {"status": "not_running", "message": "API is not currently active"}

    try:
        # Get the parent process
        parent = psutil.Process(api_server_process.pid)

        # Kill all child processes first (Uvicorn worker processes)
        for child in parent.children(recursive=True):
            child.kill()

        # Terminate the parent process
        api_server_process.terminate()

        # Wait for the process to be fully terminated with timeout
        timeout = 5  # seconds
        api_server_process.join(timeout)

        # Check if termination was successful
        if api_server_process.is_alive():
            api_server_process.kill()  # Force kill if terminate didn't work
            api_server_process.join(1)  # Give it a moment to be killed

            if api_server_process.is_alive():
                return {
                    "status": "error",
                    "message": "Failed to terminate API server process",
                }

        # Reset the global reference
        api_server_process = None
        return {"status": "success", "message": "API successfully deactivated"}

    except Exception as e:
        return {
            "status": "error",
            "message": f"Error during API deactivation: {str(e)}",
        }


def get_api_status():
    """
    Check if the API server is currently running.

    Returns:
        dict: Status information about the API server
    """
    global api_server_process

    if api_server_process is not None and api_server_process.is_alive():
        return {
            "status": "active",
            "message": f"API is active on {HOST}:{PORT}",
            "pid": api_server_process.pid,
        }
    else:
        return {"status": "inactive", "message": "API is not currently running"}
