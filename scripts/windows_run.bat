@echo off
rem Starts Ghost Image with the project virtualenv. Does not run a PowerShell script.
cd /d "%~dp0\.."
if not exist ".venv\Scripts\python.exe" (
    echo Run scripts\windows_setup.bat first.
    exit /b 1
)
".venv\Scripts\python.exe" -m ghost_image
