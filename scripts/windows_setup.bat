@echo off
rem Runs windows_setup.ps1 without requiring a changed execution policy.
cd /d "%~dp0\.."
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0windows_setup.ps1"
if errorlevel 1 exit /b 1
