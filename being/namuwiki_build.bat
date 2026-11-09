@echo off
chcp 65001 > nul
cd /d "%~dp0"
python -m pip install --user pyarrow
python namuwiki_build.py %*
pause
