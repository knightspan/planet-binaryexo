# streamlit_app.py — Root-level entrypoint for Streamlit Cloud
# This file simply re-exports dashboard/app.py so Streamlit Cloud
# can find it without needing to configure a custom main file path.
import runpy, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

runpy.run_path(str(ROOT / "dashboard" / "app.py"), run_name="__main__")
