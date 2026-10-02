@echo off
rem Starts Ghost Image with the project virtualenv. Keep this file in scripts.
rem A Desktop icon should be a shortcut to this file, not a second copy of it.
cd /d "%~dp0\.."
if not exist "config.yaml" (
    echo This file has to stay in the scripts folder next to the project.
    echo Move it back, then right-click it and choose Send to ^> Desktop ^(create shortcut^).
    pause
    exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
    echo Run scripts\windows_setup.bat first.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m ghost_image
if errorlevel 1 pause
