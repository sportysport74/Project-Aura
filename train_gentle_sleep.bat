@echo off
echo ================================================================================
echo TRAIN COSF WITH GENTLE SLEEP (OPTIMIZED)
echo ================================================================================
echo.
echo This training uses the FIXED sleep implementation:
echo   - Gentle sleep (preserves learned structure)
echo   - Lower drift threshold (1.0 instead of 2.0)
echo   - Warmup period (no sleep in epoch 1)
echo   - Configurable decay rate (0.5 = balanced)
echo.
echo Key improvements from original:
echo   1. Preserves phase relationships (semantic structure)
echo   2. Preserves chirality orientation (polarity)
echo   3. Resets fatigue (velocity, drift) without amnesia
echo   4. Expected: less than 5 percent loss spike vs 100 percent before
echo.
echo Model: TINY (256 dim, 4 layers)
echo Epochs: 3
echo Batch size: 8
echo Drift threshold: 1.0 (lowered for earlier intervention)
echo.
echo ================================================================================
echo.

cd COSF-MI

..\venv\Scripts\python.exe train_with_gentle_sleep.py --model-size tiny --epochs 3 --batch-size 8 --drift-threshold 1.0 --decay-rate 0.5

echo.
echo ================================================================================
echo TRAINING COMPLETE
echo ================================================================================
echo.
echo Checkpoints saved to: checkpoints/gentle_sleep/
echo.
echo Review the sleep cycle summary above to check:
echo   - Consolidation success rate (should be higher than 0 percent)
echo   - Loss spike after sleep (should be less than 5 percent)
echo   - Chirality preservation (balance should not drastically change)
echo.
pause
