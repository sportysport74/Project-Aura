@echo off
REM Test the trained COSF model

cd /d "%~dp0"

echo ================================================================================
echo TEST TRAINED COSF MODEL
echo ================================================================================
echo.
echo This will:
echo   1. Load your best trained model (epoch 4)
echo   2. Show model statistics (chirality, coherence, etc.)
echo   3. Generate text from test prompts
echo.
echo ================================================================================
echo.

..\venv\Scripts\python test_trained_model.py

echo.
pause
