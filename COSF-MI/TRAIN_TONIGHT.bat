@echo off
REM SIMPLE OVERNIGHT TRAINING - WORKING VERSION
REM
REM This uses the PROVEN WORKING train_with_styled_data.py
REM Just runs it for 5 epochs instead of 3

cd /d "%~dp0"

echo ================================================================================
echo OVERNIGHT TRAINING - SIMPLE AND WORKING
echo ================================================================================
echo.
echo Using the PROVEN WORKING training script (no CUDA errors)
echo.
echo Dataset: Styled TinyStories ([STORY] tokens)
echo Samples: 461k train, 24k val
echo Epochs: 5 (extended overnight)
echo Features: Chirality + Gentle Sleep
echo.
echo This is the SAME script that worked before,
echo just running longer (5 epochs instead of 3)
echo.
echo ================================================================================
echo.

..\venv\Scripts\python train_with_styled_data.py --epochs 5

echo.
echo ================================================================================
echo TRAINING COMPLETE!
echo ================================================================================
echo.
pause
