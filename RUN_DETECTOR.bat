@echo off
title Face Mask Detector - Setup & Launch
color 0A
cls
echo.
echo  ============================================================
echo   FACE MASK DETECTOR — Auto Setup ^& Launch
echo  ============================================================
echo.

:: Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo  [ERROR] Python not found. Install Python 3.8+ from python.org
    pause
    exit /b 1
)

:: Create venv if not exists
if not exist "venv\" (
    echo  [1/3] Creating virtual environment ...
    python -m venv venv
    echo        Done.
    echo.
)

:: Activate venv
call venv\Scripts\activate.bat

:: Install requirements if needed
echo  [2/3] Checking dependencies ...
pip install -q -r requirements.txt
echo        Done.
echo.

:: Run the detector
echo  [3/3] Launching detector (auto-downloads + trains if needed) ...
echo.
echo  ============================================================
echo   Controls:  Q=Quit  S=Screenshot  SPACE=Pause  +/-=Sensitivity
echo  ============================================================
echo.
python realtime_mask_detector.py %*

echo.
echo  Session ended. Press any key to exit.
pause >nul
