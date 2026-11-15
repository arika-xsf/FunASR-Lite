@echo off
REM FunASR-Lite WebUI Launcher (path-independent, auto bootstrap)
REM Usage: webui.bat               auto-detect GPU/CPU
REM        webui.bat cpu           force CPU-light install
REM        webui.bat gpu           force CUDA install
REM        webui.bat cn            use mirrors (combine: webui.bat gpu cn)
REM        FUNASR_PYTHON=[path]    optional: full path to a Python 3.10-3.13 exe
REM        FUNASR_PYPI_INDEX       optional: override the CN PyPI index URL
REM        FUNASR_TORCH_INDEX      optional: override the CN PyTorch wheel URL
setlocal
cd /d "%~dp0"

set "VENV=%~dp0venv"
set "PY=%VENV%\Scripts\python.exe"
set "MODE=GPU"
set "AUTO=1"
set "CN=0"

REM ---- parse arguments in any order ----
:args
if "%~1"=="" goto :detect
if /i "%~1"=="cpu" ( set "MODE=CPU" & set "AUTO=0" )
if /i "%~1"=="gpu" ( set "MODE=GPU" & set "AUTO=0" )
if /i "%~1"=="cn" set "CN=1"
shift
goto :args

:detect
if "%AUTO%"=="0" goto :modeinfo
where nvidia-smi >nul 2>&1
if errorlevel 1 set "MODE=CPU"

:modeinfo
if "%MODE%"=="CPU" set "REQ=%~dp0requirements-cpu.txt"
if "%MODE%"=="GPU" set "REQ=%~dp0requirements.txt"
if "%MODE%"=="GPU" (
    echo [Mode] GPU build - CUDA 12.8, NVIDIA GPU + recent driver required.
) else (
    echo [Mode] CPU-light build - no NVIDIA GPU required.
)

REM ---- a per-mode stamp marks a COMPLETED install ----
REM Checking only python.exe used to skip a retry after an interrupted first
REM install, leaving a broken venv that failed with ModuleNotFoundError.
set "STAMP=%VENV%\.installed-%MODE%"
if exist "%STAMP%" goto :run

echo [Setup] %MODE% dependencies are not installed yet.

if exist "%PY%" (
    echo [Setup] Interrupted install detected, resuming. Already installed
    echo          packages are kept - no need to delete the venv folder.
    goto :pipupgrade
)

REM ---- choose a supported base interpreter ----
REM Python 3.10-3.13 all have Windows wheels for these pinned packages.
REM Prefer the versions verified most thoroughly, then 3.12/3.13, and
REM finally fall back to whatever "python" on PATH resolves to.
set "PYCMD=python"
if defined FUNASR_PYTHON goto :custompy
py -3.11 -c "exit(0)" >nul 2>&1
if not errorlevel 1 (
    set "PYCMD=py -3.11"
    goto :pycheck
)
py -3.10 -c "exit(0)" >nul 2>&1
if not errorlevel 1 (
    set "PYCMD=py -3.10"
    goto :pycheck
)
py -3.12 -c "exit(0)" >nul 2>&1
if not errorlevel 1 (
    set "PYCMD=py -3.12"
    goto :pycheck
)
py -3.13 -c "exit(0)" >nul 2>&1
if not errorlevel 1 set "PYCMD=py -3.13"
goto :pycheck

:custompy
set PYCMD="%FUNASR_PYTHON%"

:pycheck
%PYCMD% -c "import sys; print('[Python]', sys.version.split()[0]); sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,13) else 1)"
if errorlevel 1 (
    echo ERROR: Python 3.10 - 3.13 is required.
    echo        The interpreter shown above is outside the supported range.
    echo        Install Python 3.10-3.13 from https://www.python.org/downloads/windows/
    echo        and tick "Add Python to PATH", or set FUNASR_PYTHON to its full path.
    pause
    exit /b 1
)

echo [Setup] Creating virtual environment ...
%PYCMD% -m venv "%VENV%"
if errorlevel 1 (
    echo ERROR: failed to create venv. Check the Python installation and free disk space.
    pause
    exit /b 1
)

:pipupgrade
REM ---- CN mirror defaults (keep in sync with cn_mirror.py) ----
set "CN_PYPI=https://mirrors.aliyun.com/pypi/simple"
if defined FUNASR_PYPI_INDEX set "CN_PYPI=%FUNASR_PYPI_INDEX%"
set "MIRROR_OPT="
if "%CN%"=="1" set "MIRROR_OPT=-i %CN_PYPI%"
echo [Setup] Upgrading pip ...
"%PY%" -m pip install --upgrade pip %MIRROR_OPT% --timeout 60 --retries 5
if errorlevel 1 echo WARNING: pip upgrade failed; continuing with the bundled pip.

REM ---- optional China mirrors: build a temp requirements file in venv/ ----
REM Only the index lines are swapped; the pinned versions stay single-sourced
REM in requirements*.txt and no tracked file is modified.
if "%CN%"=="0" goto :mirrordone
set "CNREQ=%VENV%\requirements-cn-%MODE%.txt"
"%PY%" "%~dp0cn_mirror.py" "%REQ%" "%CNREQ%" "%MODE%"
if errorlevel 1 (
    echo ERROR: failed to build the CN mirror requirements file.
    pause
    exit /b 1
)
set "REQ=%CNREQ%"
echo [Mirror] CN mode enabled - active index URLs are listed in %CNREQ%.
:mirrordone

if "%MODE%"=="GPU" (
    echo [Setup] Installing dependencies. The GPU build downloads a ~3.3GB torch
    echo          wheel; on a slow link this can take a long time. Do not close
    echo          this window. Progress and errors are printed by pip above.
) else (
    echo [Setup] Installing dependencies. The CPU build downloads about 200MB
    echo          of PyTorch wheels. Do not close this window.
)
"%PY%" -m pip install --timeout 60 --retries 5 -r "%REQ%"
if errorlevel 1 (
    echo.
    echo ERROR: dependency install failed. Read the pip error above.
    echo        - A network timeout on the big PyTorch wheel is the usual cause.
    echo        - Just run webui.bat again: it RESUMES and keeps every package
    echo          already installed.
    echo        - Slow overseas downloads: run  webui.bat cn
    echo          or  webui.bat gpu cn  /  webui.bat cpu cn
    echo        - Last resort: delete the venv folder and run webui.bat again.
    pause
    exit /b 1
)

echo [Setup] Verifying imports ...
"%PY%" -c "import gradio, funasr, torch" >nul 2>&1
if errorlevel 1 (
    echo ERROR: packages are on disk but fail to import.
    echo        Delete the venv folder and run webui.bat again for a clean reinstall.
    pause
    exit /b 1
)

echo %MODE%>"%STAMP%"
REM Switching modes must invalidate the other mode's stamp, otherwise running
REM the old mode again would skip the torch reinstall and keep the wrong wheel.
if "%MODE%"=="GPU" (
    if exist "%VENV%\.installed-CPU" del /q "%VENV%\.installed-CPU"
) else (
    if exist "%VENV%\.installed-GPU" del /q "%VENV%\.installed-GPU"
)
echo [Setup] Done - this one-time setup will not run again.

:run
echo [Ready] Starting WebUI at http://127.0.0.1:7860 ...
"%PY%" "%~dp0webui.py"
if errorlevel 1 (
    echo.
    echo ERROR: WebUI exited with an error. See the message above.
    echo        If a package is missing, delete venv\.installed-%MODE%
    echo        and run webui.bat again to repair the environment.
)
pause
