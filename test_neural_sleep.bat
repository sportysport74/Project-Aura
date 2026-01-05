@echo off
echo ================================================================================
echo NEURAL SLEEP FATIGUE TEST
echo ================================================================================
echo.
echo This test validates that homeostatic reset ("neural sleep") provides
echo memory consolidation benefit during training.
echo.
echo Test protocol:
echo   1. Train for 1,000 steps WITHOUT sleep
echo   2. Measure homeostatic drift
echo   3. Trigger neural sleep (reset to phi state)
echo   4. Train for 50 more steps
echo   5. Check if loss improved beyond pre-sleep minimum
echo.
echo Expected: Loss spike after sleep, then converge to lower minimum
echo          = Memory consolidation confirmed
echo.
echo ================================================================================
echo.

cd COSF-MI

..\venv\Scripts\python.exe test_neural_sleep.py

echo.
echo ================================================================================
echo TEST COMPLETE
echo ================================================================================
echo.
echo Results saved to: neural_sleep_test_results.pt
echo.

pause
