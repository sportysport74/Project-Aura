@echo off
REM Training aquarium - watch the fish while COSF trains

cd /d "%~dp0"

..\venv\Scripts\python fish.py %*
