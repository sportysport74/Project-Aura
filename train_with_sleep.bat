@echo off
echo ================================================================================
echo TRAIN COSF OSCILLATORY MODEL WITH AUTOMATIC SLEEP CYCLES
echo ================================================================================
echo.
echo This trains the oscillatory transformer with automatic homeostatic reset
echo when the model becomes "fatigued" (high drift from phi).
echo.
echo Features:
echo   - Automatic sleep when drift ^> 2.0
echo   - Memory consolidation tracking
echo   - Sleep cycle statistics
echo   - Model learns WHEN to rest, not just HOW to learn
echo.
echo Model: TINY (256 dim, 4 layers) - safe settings for RTX 5090
echo Epochs: 2
echo Batch size: 8 (conservative)
echo Sequence length: 256 (reduced from 512)
echo.
echo ================================================================================
echo.

cd COSF-MI

..\venv\Scripts\python.exe train_with_sleep.py --model-size tiny --epochs 2 --batch-size 8 --drift-threshold 2.0

echo.
echo ================================================================================
echo TRAINING COMPLETE
echo ================================================================================
echo.
echo Checkpoints saved to: checkpoints/cosf_oscillatory/
echo.
echo Sleep cycle summary was printed above.
echo Check if memory consolidation occurred after sleep events!
echo.

pause
