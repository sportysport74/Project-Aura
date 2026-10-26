@echo off
REM Continue Training from Epoch 4 Checkpoint
REM
REM This resumes training from your best checkpoint without starting over

cd /d "%~dp0"

echo ================================================================================
echo CONTINUE TRAINING - RESUME FROM EPOCH 4
echo ================================================================================
echo.
echo This will:
echo   1. Load your epoch 4 checkpoint (val_loss: 6.2987)
echo   2. Resume training from epoch 5
echo   3. Train for 5 MORE epochs (epochs 5-9)
echo   4. Keep all your existing progress
echo.
echo Your previous training:
echo   - Epochs 1-4 completed
echo   - Best val loss: 6.2987
echo   - Best perplexity: 543.88
echo   - Chirality balance: 50.3%%
echo.
echo ================================================================================
echo.

..\venv\Scripts\python continue_training.py --resume checkpoints/styled_cosf/checkpoint_epoch_4.pt --epochs 5

echo.
echo ================================================================================
echo CONTINUED TRAINING COMPLETE!
echo ================================================================================
echo.
pause
