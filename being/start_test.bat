@echo off
chcp 65001 > nul
cd /d "%~dp0"
rem 진짜 존재와 따로 도는 시험용 존재. test.db와 letters_test를 쓰고, 알림은 끔. 페이지는 localhost:8766
python live.py --config config_test.json %*
pause
