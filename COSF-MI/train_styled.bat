@echo off
REM Train COSF with Style Token Conditioning
REM
REM This trains on TinyStories with [STORY] token prepended.
REM
REM COSF features preserved:
REM   - Oscillators
REM   - Chirality
REM   - Homeostatic sleep (gentle)
REM   - Julia bounds
REM   - Phase coherence
REM
REM NEW: Style token conditioning
REM   - Data has [STORY] token prepended
REM   - Model learns to condition on style token

cd /d "%~dp0"

echo ================================================================================
echo TRAIN COSF WITH STYLE TOKEN CONDITIONING
echo ================================================================================
echo.
echo This training uses styled TinyStories data with [STORY] token.
echo.
echo COSF features preserved:
echo   [OK] Oscillators
echo   [OK] Chirality
echo   [OK] Homeostatic sleep (gentle)
echo   [OK] Julia bounds
echo   [OK] Phase coherence
echo.
echo NEW: Style token conditioning
echo   [OK] Training on styled_train.txt
echo   [OK] Same COSF architecture (no model changes)
echo.
echo Model: TINY (256 dim, 4 layers)
echo Epochs: 3
echo Batch size: 8
echo.
echo ================================================================================
echo.

..\venv\Scripts\python train_with_styled_data.py --model-size tiny --epochs 3 --batch-size 8

pause
