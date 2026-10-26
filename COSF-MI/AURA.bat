@echo off
REM ============================================================================
REM AURA - COSF-MI Training & Testing Interface
REM ============================================================================
REM
REM Quick access to all AURA training and testing commands
REM

cd /d "%~dp0"

:MENU
cls
echo.
echo ============================================================================
echo    AURA - Adaptive Unified Resonance Architecture
echo ============================================================================
echo.
echo    Current Status:
echo      - Model: Tiny COSF (9.4M params, 256 dim, 4 layers)
echo      - Latest: Epoch 4, val_loss=6.2987, perplexity=543.88
echo      - Chirality: ENABLED (50.3%% balance)
echo      - Dataset: Styled TinyStories (461k train, 24k val)
echo.
echo ============================================================================
echo.
echo    TRAINING OPTIONS:
echo.
echo    [1] Continue Training (Resume from Epoch 4)
echo        - Loads checkpoint_epoch_4.pt
echo        - Trains 5 MORE epochs (Epochs 5-9)
echo        - Keeps all existing progress
echo        - Time: ~6-8 hours
echo.
echo    [2] Start Fresh Training (3 Epochs - Quick Test)
echo        - Uses train_with_styled_data.py
echo        - 3 epochs from scratch
echo        - Time: ~2-3 hours
echo.
echo    [3] Overnight Training (5 Epochs - Extended)
echo        - Uses TRAIN_TONIGHT.bat
echo        - 5 epochs from scratch
echo        - Time: ~6-8 hours
echo.
echo ============================================================================
echo.
echo    TESTING OPTIONS:
echo.
echo    [4] Chat with AURA (Interactive Conversation)
echo        - Have a back-and-forth conversation
echo        - Type prompts and get responses
echo        - Adjustable temperature and settings
echo.
echo    [5] Test Current Model
echo        - Loads best_model.pt (Epoch 4)
echo        - Shows model statistics
echo        - Generates text from prompts
echo.
echo    [6] Analyze Training Results
echo        - Shows training history
echo        - Compares checkpoints
echo        - Displays metrics evolution
echo.
echo ============================================================================
echo.
echo    OTHER OPTIONS:
echo.
echo    [7] Check GPU Status
echo        - Show CUDA availability
echo        - Display GPU memory
echo.
echo    [8] View Project Status
echo        - Show current project state
echo        - List available checkpoints
echo.
echo    [0] Exit
echo.
echo ============================================================================
echo.

set /p choice="Select option [0-8]: "

if "%choice%"=="1" goto CONTINUE
if "%choice%"=="2" goto FRESH
if "%choice%"=="3" goto OVERNIGHT
if "%choice%"=="4" goto CHAT
if "%choice%"=="5" goto TEST
if "%choice%"=="6" goto ANALYZE
if "%choice%"=="7" goto GPU
if "%choice%"=="8" goto STATUS
if "%choice%"=="0" goto END

echo Invalid choice. Press any key to try again...
pause >nul
goto MENU

:CONTINUE
cls
echo.
echo ============================================================================
echo CONTINUE TRAINING FROM EPOCH 4
echo ============================================================================
echo.
call continue_training.bat
pause
goto MENU

:FRESH
cls
echo.
echo ============================================================================
echo START FRESH TRAINING (3 EPOCHS)
echo ============================================================================
echo.
echo WARNING: This will start training from scratch (not using your checkpoint)
echo Press Ctrl+C to cancel, or any key to continue...
pause >nul
..\venv\Scripts\python train_with_styled_data.py --epochs 3
pause
goto MENU

:OVERNIGHT
cls
echo.
echo ============================================================================
echo OVERNIGHT TRAINING (5 EPOCHS FROM SCRATCH)
echo ============================================================================
echo.
echo WARNING: This will start training from scratch (not using your checkpoint)
echo Press Ctrl+C to cancel, or any key to continue...
pause >nul
call TRAIN_TONIGHT.bat
pause
goto MENU

:CHAT
cls
echo.
echo ============================================================================
echo CHAT WITH AURA
echo ============================================================================
echo.
call chat.bat
goto MENU

:TEST
cls
echo.
echo ============================================================================
echo TEST CURRENT MODEL
echo ============================================================================
echo.
call test_model.bat
pause
goto MENU

:ANALYZE
cls
echo.
echo ============================================================================
echo TRAINING RESULTS ANALYSIS
echo ============================================================================
echo.
echo Opening TRAINING_RESULTS_ANALYSIS.md...
echo.
type TRAINING_RESULTS_ANALYSIS.md | more
echo.
echo.
echo Press any key to continue...
pause >nul
goto MENU

:GPU
cls
echo.
echo ============================================================================
echo GPU STATUS CHECK
echo ============================================================================
echo.
..\venv\Scripts\python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}') if torch.cuda.is_available() else None; print(f'GPU Name: {torch.cuda.get_device_name(0)}') if torch.cuda.is_available() else None; print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB') if torch.cuda.is_available() else None"
echo.
pause
goto MENU

:STATUS
cls
echo.
echo ============================================================================
echo PROJECT STATUS
echo ============================================================================
echo.
echo Current Directory: %CD%
echo.
echo.
echo Available Checkpoints:
echo ----------------------
dir /b checkpoints\styled_cosf\*.pt 2>nul
if errorlevel 1 echo No checkpoints found!
echo.
echo.
echo Recent Training Files:
echo ----------------------
dir /b /o-d *.py 2>nul | findstr /i "train test"
echo.
echo.
echo Key Scripts:
echo ------------
echo   - continue_training.py  : Resume from checkpoint
echo   - train_with_styled_data.py : Fresh training (proven working)
echo   - test_trained_model.py : Test best model
echo   - TRAIN_TONIGHT.bat     : 5 epoch overnight training
echo.
echo.
echo Dataset Status:
echo ---------------
if exist "data\tinystories\styled_train.txt" (
    echo   [OK] styled_train.txt found
) else (
    echo   [MISSING] styled_train.txt
)
if exist "data\tinystories\styled_val.txt" (
    echo   [OK] styled_val.txt found
) else (
    echo   [MISSING] styled_val.txt
)
echo.
echo.
echo Tokenizer:
echo ----------
if exist "models\tokenizer_8k.model" (
    echo   [OK] tokenizer_8k.model (vocab=8000)
) else (
    echo   [MISSING] tokenizer_8k.model
)
echo.
pause
goto MENU

:END
cls
echo.
echo ============================================================================
echo Thanks for using AURA!
echo ============================================================================
echo.
echo Remember: Your best model is saved at:
echo   checkpoints/styled_cosf/best_model.pt
echo.
echo To resume training later, just run:
echo   continue_training.bat
echo.
exit /b
