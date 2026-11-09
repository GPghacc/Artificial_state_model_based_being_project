@echo off
chcp 65001 > nul
cd /d "%~dp0"
title 판단 기관 (d1-3B)
.venv-decision\Scripts\python decision_server.py %*
pause
