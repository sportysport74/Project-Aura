@echo off
REM OVERNIGHT TRAINING: AURA LEVEL 1 + BRIDGE WORDS
REM
REM This trains on multi-intent data ([STORY] + [CODE]) with semantic bridge words.
REM Extended training (5 epochs) to observe emergent chirality differentiation.
REM
REM Bridge words that force semantic disambiguation:
REM   - buffer: story=protect, code=memory
REM   - branch: story=tree, code=git/conditional
REM   - loop: story=circle, code=iteration
REM   - class: story=school, code=OOP
REM   - root: story=plant, code=directory
REM
REM Expected results by morning:
REM   - Coherence rising from 0.01 to 0.20+
REM   - Chirality flips beginning to occur
REM   - Semantic differentiation on bridge words
REM   - 80%+ consolidation success rate
REM
REM Training will save checkpoints every epoch to:
REM   checkpoints/overnight_bridge_words/
REM
REM Run analyze_morning.bat when you wake up to see results!

cd /d "%~dp0"

echo ================================================================================
echo OVERNIGHT TRAINING: AURA LEVEL 1 + BRIDGE WORDS
echo ================================================================================
echo.
echo Starting extended overnight training...
echo.
echo Dataset: Multi-Intent ([STORY] + [CODE])
echo Bridge words: buffer, branch, loop, class, root
echo Epochs: 5 (extended overnight)
echo Warmup: 45k steps (heartbeat formation)
echo.
echo AURA LEVEL 1 features:
echo   - Kuramoto coherence loss
echo   - Drift-gated coupling
echo   - Coherence tracking
echo.
echo Expected by morning:
echo   - Coherence: 0.01 -^> 0.20+
echo   - Chirality flips on bridge words
echo   - Semantic differentiation!
echo.
echo Checkpoints: checkpoints/overnight_bridge_words/
echo.
echo ================================================================================
echo.
echo Training starting... Go game and sleep well!
echo Run analyze_morning.bat when you wake up to see results.
echo.

..\venv\Scripts\python train_overnight.py

echo.
echo ================================================================================
echo TRAINING COMPLETE!
echo ================================================================================
echo.
echo Run: analyze_morning.bat
echo   to see coherence evolution, chirality metrics, and results!
echo.
pause
