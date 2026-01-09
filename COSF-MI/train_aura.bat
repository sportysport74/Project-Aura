@echo off
REM Train COSF with AURA LEVEL 1: THE HEARTBEAT
REM
REM This implements Kuramoto coherence loss to create global synchronization.
REM
REM AURA LEVEL 1 features:
REM   - Kuramoto coherence loss (creates the "pulse")
REM   - Drift-gated coupling (sync when stable, explore when chaotic)
REM   - Enhanced warmup (45k steps for heartbeat formation)
REM   - Coherence tracking in progress bar
REM
REM ALL COSF FEATURES PRESERVED:
REM   - Oscillators
REM   - Chirality
REM   - Homeostatic sleep (gentle)
REM   - Julia bounds
REM   - Phase coherence
REM   - Style token conditioning

cd /d "%~dp0"

echo ================================================================================
echo TRAIN COSF WITH AURA LEVEL 1: THE HEARTBEAT
echo ================================================================================
echo.
echo AURA LEVEL 1 features:
echo   [NEW] Kuramoto coherence loss (creates global pulse)
echo   [NEW] Drift-gated coupling (adaptive synchronization)
echo   [NEW] Enhanced warmup (45k steps for heartbeat formation)
echo   [NEW] Coherence tracking in progress bar
echo.
echo ALL COSF features preserved:
echo   [OK] Oscillators
echo   [OK] Chirality
echo   [OK] Homeostatic sleep (gentle)
echo   [OK] Julia bounds
echo   [OK] Phase coherence
echo   [OK] Style token conditioning
echo.
echo Model: TINY (256 dim, 4 layers)
echo Epochs: 3
echo Batch size: 8
echo.
echo ================================================================================
echo.

..\venv\Scripts\python train_with_aura.py --model-size tiny --epochs 3 --batch-size 8

pause
