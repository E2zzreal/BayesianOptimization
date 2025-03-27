@echo off
echo ===== Bayesian Optimization Material Composition System Packager =====
echo.

REM Check Python installation
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Error: Python not detected. Please install Python 3.9 or higher.
    pause
    exit /b 1
)

REM Create and activate virtual environment
echo [1/5] Creating virtual environment...
if exist venv (
    echo Deleting existing virtual environment...
    rmdir /s /q venv
)
python -m venv venv
if %ERRORLEVEL% neq 0 (
    echo Error: Failed to create virtual environment
    pause
    exit /b 1
)

call venv\Scripts\activate
if %ERRORLEVEL% neq 0 (
    echo Error: Failed to activate virtual environment
    pause
    exit /b 1
)

REM Install dependencies
echo [2/5] Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install cx_Freeze
if %ERRORLEVEL% neq 0 (
    echo Error: Dependency installation failed
    pause
    exit /b 1
)

REM Create data directory
echo [3/5] Preparing data directory...
if not exist data mkdir data

REM Build application
echo [4/5] Building application...
python setup.py build_exe
if %ERRORLEVEL% neq 0 (
    echo Error: Application build failed
    pause
    exit /b 1
)

REM Create installer package
echo [5/5] Creating installer package...
python setup.py bdist_msi
if %ERRORLEVEL% neq 0 (
    echo Error: Installer creation failed
    pause
    exit /b 1
)

echo.
echo Build completed successfully!
echo Executable location: build\exe.win-amd64-3.9\BayesianOptimizationSystem.exe
echo Installer location: dist\BayesianOptimizationSystem-1.0.0.msi

REM Deactivate virtual environment
call venv\Scripts\deactivate.bat

pause