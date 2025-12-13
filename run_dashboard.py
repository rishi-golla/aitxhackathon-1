#!/usr/bin/env python3
"""
Run the OSHA Vision Dashboard.

Usage:
    python run_dashboard.py
    # or
    streamlit run src/dashboard/app.py
"""

import subprocess
import sys


def main():
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        "src/dashboard/app.py",
        "--server.address", "0.0.0.0",
        "--server.port", "8501"
    ]

    print("Starting OSHA Vision Dashboard...")
    print("Dashboard will be available at: http://localhost:8501")
    print()

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
