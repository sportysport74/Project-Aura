@echo off
REM Chat with AURA - Interactive Conversation

cd /d "%~dp0"

echo ================================================================================
echo CHAT WITH AURA
echo ================================================================================
echo.
echo Have an interactive conversation with your trained model!
echo.
echo The model will respond to your prompts in real-time.
echo.
echo Tips:
echo   - Just talk to AURA in simple sentences (type reset for a new chat)
echo   - For a story model: chat.bat checkpoints\fixed_run_100ep\best_model.pt
echo   - Type 'quit' or 'exit' to end the chat
echo   - Type 'stats' to see model metrics
echo   - Type 'temp 0.7' to adjust temperature
echo.
echo ================================================================================
echo.

..\venv\Scripts\python chat_with_aura.py %*

echo.
echo ================================================================================
echo CHAT COMPLETE
echo ================================================================================
echo.
pause
