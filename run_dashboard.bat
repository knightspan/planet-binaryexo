@echo off
REM Launch the Streamlit dashboard (fixes Windows PATH issue)
call .venv\Scripts\activate.bat
echo Starting JyotirVega Dashboard...
echo Open http://localhost:8501 in your browser
echo Press Ctrl+C to stop
echo.
python -m streamlit run dashboard/app.py
