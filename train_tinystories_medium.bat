@echo off
REM Train COSF Medium Model (100M params) on TinyStories
REM This will take ~6-10 hours on RTX 5090
REM Safe to interrupt - will save checkpoints every epoch

echo ========================================================================
echo COSF TINYSTORIES TRAINING - MEDIUM MODEL (100M PARAMS)
echo ========================================================================
echo.
echo Training will start in 5 seconds...
echo Press Ctrl+C now to cancel, or wait to continue
echo.
timeout /t 5

cd /d "C:\Users\fatba\COSF-MI\COSF-MI"

echo.
echo Starting training...
echo.

..\venv\Scripts\python.exe -m cosf_mi.training.train_cosf_large --model-size medium --data-dir data/tinystories --tokenizer models/tokenizer_8k.model --epochs 20 --device cuda

echo.
echo ========================================================================
echo TRAINING COMPLETE!
echo ========================================================================
echo.
echo Next steps:
echo   1. Check: checkpoints\cosf_large\best_model.pt
echo   2. Copy to: checkpoints\cosf_transformer\best_model.pt
echo   3. Run: python jarvis_cli.py
echo.
pause
