@echo off
REM Train the CNN on synthetic data
call .venv\Scripts\activate.bat
echo Training JyotirVega DualViewCNN...
echo.
python train_model.py --synthetic --epochs 50 --n-per-class 300 --eval --calibrate
echo.
echo Training complete! Model saved to models/classifier.h5
pause
