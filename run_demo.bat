@echo off
REM Run synthetic demo — no internet needed
call .venv\Scripts\activate.bat
echo Running JyotirVega Pipeline Demo...
echo.
python run_pipeline.py --demo --no-fap
echo.
echo Done! Check outputs/ folder for plots and report.html
pause
