@echo off
REM FunASR-Lite WebUI Launcher (path-independent, auto bootstrap)
REM Usage: webui.bat        auto-detect GPU
REM        webui.bat cpu    force CPU-light install
REM        webui.bat gpu    force CUDA install
setlocal
cd /d "%~dp0"
set "VENV=%~dp0venv"
set "PY=%VENV%\Scripts\python.exe"
set "REQ=%~dp0requirements.txt"
set "MODE=GPU"

if /i "%~1"=="cpu" goto :usecpu
if /i "%~1"=="gpu" goto :usegpu
where nvidia-smi >nul 2>&1
if errorlevel 1 goto :usecpu

:usegpu
echo [Mode] GPU build - CUDA 12.8, NVIDIA GPU + recent driver required.
goto :install

:usecpu
set "REQ=%~dp0requirements-cpu.txt"
set "MODE=CPU"
echo [Mode] CPU-light build - no NVIDIA GPU required.

:install
if exist "%PY%" goto :run
echo [First run] Creating virtual environment for %MODE% build...
echo [First run] Installing dependencies, this may take several minutes.
python -m venv "%VENV%"
if errorlevel 1 (
    echo ERROR: python not found. Install Python 3.10/3.11 and add to PATH, then re-run.
    pause
    exit /b 1
)
"%PY%" -m pip install -r "%REQ%"
if errorlevel 1 (
    echo ERROR: dependency install failed. Check internet and the requirements file.
    pause
    exit /b 1
)

:run
"%PY%" "%~dp0webui.py"
pause
