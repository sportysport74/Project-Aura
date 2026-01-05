@echo off
REM COSF-MI CUDA Build Script for Windows
REM Requires NVIDIA CUDA Toolkit installed

echo Building COSF-MI CUDA kernels...

REM Check for nvcc
where nvcc >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: nvcc not found. Please install CUDA Toolkit and add to PATH.
    exit /b 1
)

REM Create output directory
if not exist "build" mkdir build

REM Compile phi_decomp.cu (RTX 5090: Blackwell sm_90)
echo Compiling phi_decomp.cu...
nvcc -c phi_decomp.cu -o build/phi_decomp.o ^
    -O3 ^
    --use_fast_math ^
    -arch=sm_90 ^
    -Xcompiler "/MD"

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to compile phi_decomp.cu
    exit /b 1
)

REM Compile julia_stability.cu (RTX 5090: Blackwell sm_90)
echo Compiling julia_stability.cu...
nvcc -c julia_stability.cu -o build/julia_stability.o ^
    -O3 ^
    --use_fast_math ^
    -arch=sm_90 ^
    -Xcompiler "/MD"

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to compile julia_stability.cu
    exit /b 1
)

REM Create shared library (RTX 5090: Blackwell sm_90)
echo Creating shared library...
nvcc -shared ^
    build/phi_decomp.o ^
    build/julia_stability.o ^
    -o build/cosf_cuda.dll ^
    -arch=sm_90

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to create shared library
    exit /b 1
)

echo.
echo Build successful!
echo Output: build/cosf_cuda.dll
echo.
echo To use with Python:
echo   import ctypes
echo   lib = ctypes.CDLL('./build/cosf_cuda.dll')
