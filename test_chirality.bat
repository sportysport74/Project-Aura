@echo off
echo ================================================================================
echo TEST CHIRALITY FEATURES - COSF OSCILLATOR
echo ================================================================================
echo.
echo This tests the new chirality (spiral handedness) features:
echo   - ChiralOscillatorLayer: Left/right handed spirals
echo   - SemanticPolarityDetector: Learns positive/negative concepts
echo   - ChiralAttention: Attention heads with alternating handedness
echo   - Geometric semantic differentiation
echo   - Terahertz scaling for future hardware
echo.
echo Based on MIT's p-wave magnetism research
echo ================================================================================
echo.

cd COSF-MI

..\venv\Scripts\python.exe test_chirality.py

echo.
echo ================================================================================
echo TEST COMPLETE
echo ================================================================================
echo.
pause
