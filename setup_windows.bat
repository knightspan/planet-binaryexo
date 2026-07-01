@echo off
REM ═══════════════════════════════════════════════════════════════════════
REM  JyotirVega Exoplanet Pipeline — Windows Setup Script
REM  ISRO BAH 2026 · Problem Statement 7 · Team JyotirVega · Aurixys
REM
REM  Run this ONCE after extracting the project.
REM  Double-click this file or run from command prompt.
REM ═══════════════════════════════════════════════════════════════════════

echo.
echo ╔══════════════════════════════════════════════════════════╗
echo ║  JyotirVega Exoplanet Pipeline — Windows Setup          ║
echo ║  ISRO BAH 2026 · PS-7 · Team JyotirVega · Aurixys       ║
echo ╚══════════════════════════════════════════════════════════╝
echo.

REM Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found. Install from https://python.org
    pause
    exit /b 1
)
echo [OK] Python found

REM Create virtual environment
echo.
echo [1/5] Creating virtual environment (.venv)...
python -m venv .venv
if %errorlevel% neq 0 (
    echo ERROR: Failed to create virtual environment.
    pause
    exit /b 1
)

REM Activate
call .venv\Scripts\activate.bat

REM Upgrade pip
echo [2/5] Upgrading pip...
python -m pip install --upgrade pip --quiet

REM Install all dependencies
echo [3/5] Installing all dependencies (3-8 minutes)...
echo       transitleastsquares wotan batman lightkurve tensorflow streamlit ...
pip install -r requirements.txt --quiet
if %errorlevel% neq 0 (
    echo WARNING: Some packages may have failed. Trying individually...
    pip install transitleastsquares wotan batman-package lightkurve astropy astroquery --quiet
    pip install emcee corner tensorflow scikit-learn numpy pandas matplotlib --quiet
    pip install streamlit scipy requests --quiet
)

REM Verify key packages
echo [4/5] Verifying installation...
python -c "import transitleastsquares; print('  [OK] TLS')" 2>nul || echo "  [WARN] TLS not installed — BLS fallback will be used"
python -c "import wotan; print('  [OK] Wotan')" 2>nul || echo "  [WARN] Wotan — SG fallback will be used"
python -c "import batman; print('  [OK] Batman')" 2>nul || echo "  [WARN] Batman — MAP fitting unavailable"
python -c "import streamlit; print('  [OK] Streamlit')" 2>nul || echo "  [WARN] Streamlit"
python -c "import tensorflow; print('  [OK] TensorFlow')" 2>nul || echo "  [WARN] TensorFlow — rule-based classifier only"
python -c "import lightkurve; print('  [OK] Lightkurve')" 2>nul || echo "  [WARN] Lightkurve — real TESS download unavailable"

REM Create directories
echo [5/5] Creating project directories...
mkdir outputs 2>nul
mkdir models 2>nul
mkdir data\cache 2>nul
mkdir data\training 2>nul
mkdir data\validation 2>nul
mkdir scripts 2>nul

echo.
echo ╔══════════════════════════════════════════════════════════╗
echo ║  Setup complete! Now run one of these:                  ║
echo ║                                                          ║
echo ║  1. Quick demo (no internet):                            ║
echo ║     run_demo.bat                                         ║
echo ║                                                          ║
echo ║  2. Interactive dashboard:                               ║
echo ║     run_dashboard.bat                                    ║
echo ║                                                          ║
echo ║  3. Train the CNN:                                       ║
echo ║     run_train.bat                                        ║
echo ║                                                          ║
echo ║  4. Real TESS target:                                    ║
echo ║     .venv\Scripts\python run_pipeline.py --tics 261136679║
echo ╚══════════════════════════════════════════════════════════╝
echo.
pause
