@echo off
REM Train COSF Small Model (72M params) on TinyStories
REM OOM fixed: batch 16 + gradient accumulation 4 = effective batch 64
REM 10 epochs for faster training (can resume for more later)
REM Safe to interrupt - will save checkpoints every epoch

echo ========================================================================
echo COSF TINYSTORIES TRAINING - SMALL MODEL (72M PARAMS)
echo ========================================================================
echo.
echo Settings:
echo   - Batch size: 16 (with 4x gradient accumulation = 64 effective)
echo   - Epochs: 10 (reduced for faster training)
echo   - Memory usage: ~22-25 GB (fits in RTX 5090)
echo.
echo Training will start in 5 seconds...
echo Press Ctrl+C now to cancel, or wait to continue
echo.
timeout /t 5

cd /d "C:\Users\fatba\COSF-MI\COSF-MI"

echo.
echo Starting training...
echo.

..\venv\Scripts\python.exe -m cosf_mi.training.train_cosf_large --model-size small --data-dir data/tinystories --tokenizer models/tokenizer_8k.model --epochs 10 --device cuda

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
