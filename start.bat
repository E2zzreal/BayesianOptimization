@echo off

REM Check if Python is installed
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Python is not installed. Please install Python 3.9 or later.
    pause
    exit /b 1
)

REM Check and install dependencies
echo Installing required dependencies...
pip install -r requirements.txt

REM Set environment variables for Docker compatibility
set PYTHONPATH=%~dp0
set STREAMLIT_SERVER_PORT=8501
set STREAMLIT_SERVER_HEADLESS=true
set STREAMLIT_ARROW_DATAFRAME_USE_LEGACY=true

REM Start the application
echo Starting Bayesian Optimization Material System...
streamlit run app/main.py

pause