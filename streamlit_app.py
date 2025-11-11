import os
import sys
import runpy

# Resolve path to the dashboard app
HERE = os.path.dirname(__file__)
APP_DIR = os.path.join(HERE, "scripts", "dashboard")
APP_PATH = os.path.join(APP_DIR, "app.py")

# Ensure the app directory is importable and run the app
sys.path.insert(0, APP_DIR)
if __name__ == "__main__":
    runpy.run_path(APP_PATH, run_name="__main__")
