@echo off
cd /d C:\Users\sanvi\OneDrive\Desktop\layout
venv\Scripts\python.exe generate_use_case_docx.py > C:\Users\sanvi\docx_log.txt 2>&1
echo Exit: %errorlevel% >> C:\Users\sanvi\docx_log.txt
