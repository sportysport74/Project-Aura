@echo off
REM Morning Analysis: Check overnight training results

cd /d "%~dp0"

echo ================================================================================
echo GOOD MORNING! Let's see what happened overnight...
echo ================================================================================
echo.

..\venv\Scripts\python analyze_morning.py

echo.
echo ================================================================================
echo.
echo Check the analysis above for:
echo   1. Coherence evolution (did it rise to 0.20+?)
echo   2. Chirality flips (are they happening?)
echo   3. Consolidation success rate (^>50%%?)
echo   4. Training stability (no crashes?)
echo.
echo ================================================================================
echo.

pause
