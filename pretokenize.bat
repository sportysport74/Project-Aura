@echo off
REM Pre-tokenize TinyStories data for instant training startup
REM Run this once, then training will start immediately!

echo ========================================================================
echo PRE-TOKENIZE TINYSTORIES DATA
echo ========================================================================
echo.
echo This will tokenize and cache train.txt and val.txt
echo After completion, training will start instantly (no waiting!)
echo.
echo Press Ctrl+C to cancel, or wait 3 seconds to continue...
timeout /t 3

cd /d "C:\Users\fatba\COSF-MI\COSF-MI"

echo.
echo Starting pre-tokenization...
echo.

..\venv\Scripts\python.exe pretokenize_data.py

echo.
echo ========================================================================
echo DONE!
echo ========================================================================
echo.
echo Now you can start training and it will be instant:
echo   train_tinystories_small.bat
echo   OR
echo   train_tinystories_medium.bat
echo.
pause
