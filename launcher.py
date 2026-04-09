"""Occursus Benchmark Launcher — double-click to start the server and open the UI."""

import os
import sys
import threading
import time
import webbrowser

# ── Path setup (works both as .exe and as .py) ──
if getattr(sys, "frozen", False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

os.chdir(BASE_DIR)
sys.path.insert(0, BASE_DIR)

PORT = 8000
URL = f"http://localhost:{PORT}"

BANNER = r"""
   ___                                    ____                  _                          _
  / _ \  ___ ___ _   _ _ __ ___ _   _ ___| __ )  ___ _ __   ___| |__  _ __ ___   __ _ _ __| | __
 | | | |/ __/ __| | | | '__/ __| | | / __|  _ \ / _ \ '_ \ / __| '_ \| '_ ` _ \ / _` | '__| |/ /
 | |_| | (_| (__| |_| | |  \__ \ |_| \__ \ |_) |  __/ | | | (__| | | | | | | | | (_| | |  |   <
  \___/ \___\___|\__,_|_|  |___/\__,_|___/____/ \___|_| |_|\___|_| |_|_| |_| |_|\__,_|_|  |_|\_\

  Multi-LLM Synthesis Benchmark
"""


def open_browser():
    time.sleep(2.5)
    webbrowser.open(URL)


def main():
    print(BANNER)
    print(f"  Starting server at {URL}")
    print(f"  Working directory: {BASE_DIR}")
    print()
    print("  Close this window to stop the server.")
    print("  " + "=" * 50)
    print()

    # Open browser in background thread
    threading.Thread(target=open_browser, daemon=True).start()

    # Import app directly (avoids string-import issues in frozen .exe)
    from app import app as application
    import uvicorn

    try:
        uvicorn.run(
            application,
            host="0.0.0.0",
            port=PORT,
            log_level="info",
            access_log=False,
        )
    except KeyboardInterrupt:
        print("\n  Server stopped.")


if __name__ == "__main__":
    main()
