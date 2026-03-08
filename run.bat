@echo off
REM Change to the directory of this script
cd /d "%~dp0"

REM Activate the virtual environment
call .venv\Scripts\activate

REM Run the Python script
python clip_search_gui.py

REM Pause so the window doesn't close immediately
pause
