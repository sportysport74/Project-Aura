@echo off
echo ================================================================================
echo TRAIN COSF OSCILLATORY MODEL WITH CHIRALITY + SLEEP CYCLES
echo ================================================================================
echo.
echo This trains the oscillatory transformer with:
echo   - Chirality (spiral handedness for semantic differentiation)
echo   - Automatic sleep cycles when fatigued (drift over 2.0)
echo   - Memory consolidation tracking
echo.
echo Chirality Features:
echo   - Right-handed spirals: Positive concepts (good, happy, success)
echo   - Left-handed spirals: Negative concepts (bad, sad, failure)
echo   - Model learns to flip perspective based on context
echo.
echo Model: TINY (256 dim, 4 layers) - safe settings for RTX 5090
echo Epochs: 2
echo Batch size: 8 (conservative)
echo Sequence length: 256 (reduced from 512)
echo.
echo ================================================================================
echo.

cd COSF-MI

..\venv\Scripts\python.exe train_with_chirality.py --model-size tiny --epochs 2 --batch-size 8 --drift-threshold 2.0

echo.
echo ================================================================================
echo TRAINING COMPLETE
echo ================================================================================
echo.
echo Checkpoints saved to: checkpoints/cosf_oscillatory_chiral/
echo.
echo Next steps:
echo   - Check chirality statistics in output
echo   - Analyze which tokens learned positive vs negative handedness
echo   - Test semantic differentiation on inference
echo.
pause
