@echo off
cd /d "%~dp0"
echo Starting TM16 Visualizer...
call conda run --live-stream --name lidar_learn python main.py %*
if %ERRORLEVEL% NEQ 0 (
    python main.py %*
)
if %ERRORLEVEL% NEQ 0 (
    echo Error code: %ERRORLEVEL%
    pause
)
