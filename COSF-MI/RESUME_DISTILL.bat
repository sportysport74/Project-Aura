@echo off
REM After a restart: resume the teacher data run, the auto fine-tune chain, and the fish.
REM Safe to run any time - everything picks up where it left off.

cd /d "%~dp0"

REM Ollama (the teacher) - start it if it isn't running
curl -s http://localhost:11434 >nul 2>&1 || start "Ollama" /min ollama serve
timeout /t 5 /nobreak >nul

REM Data run (resumes from data\cosf_distill_verified.jsonl)
start "AURA data" /min cmd /c "..\venv\Scripts\python -u scripts\generate_cosf_distill.py --n 10000 > logs\distill_10k.log 2> logs\distill_10k.err.log"

REM Chain: waits for the data, stops Ollama, fine-tunes, runs the math exam
start "AURA chain" /min "C:\Program Files\Git\bin\bash.exe" run_distill_finetune.sh

timeout /t 10 /nobreak >nul
call fish.bat
