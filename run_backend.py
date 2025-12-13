#!/usr/bin/env python3
"""
Run the OSHA Vision Backend API.

Usage:
    python run_backend.py
    # or
    python server/main.py
"""

import subprocess
import sys


def main():
    print("Starting OSHA Vision Backend...")
    print("API will be available at: http://localhost:8000")
    print("Video feed: http://localhost:8000/video_feed")
    print("Status: http://localhost:8000/status")
    print()

    try:
        subprocess.run([sys.executable, "server/main.py"], check=True)
    except KeyboardInterrupt:
        print("\nBackend stopped.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
