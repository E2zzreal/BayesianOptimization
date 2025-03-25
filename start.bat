@echo off

REM Check if Python is installed
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Python is not installed. Please install Python 3.9 or later.
    pause
    exit /b 1
)

REM Check and install dependencies
pip install -r requirements.txt

REM Start the application
streamlit run app/main.py

pause