@echo off
REM Experiment tank - one fish per running job (training run + A/B arms)

cd /d "%~dp0"

..\venv\Scripts\python tank.py %*
