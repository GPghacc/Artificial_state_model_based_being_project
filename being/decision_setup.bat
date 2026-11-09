@echo off
chcp 65001 > nul
cd /d "%~dp0"
echo.
echo  판단 기관(LiquidAI d1-3B) 설치. 이 폴더 안의 .venv-decision 에만 설치되고, 컴퓨터 전체에는 아무것도 깔지 않는다.
echo  내려받는 양: PyTorch 약 3GB + 모델 약 6.3GB
echo.
python -m venv .venv-decision || goto :fail
call .venv-decision\Scripts\activate.bat
python -m pip install --upgrade pip || goto :fail
echo [1/3] PyTorch (CUDA 12.8, RTX 50 시리즈용)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 || goto :fail
echo [2/3] transformers
pip install "transformers>=5.14" pillow accelerate huggingface_hub || goto :fail
echo [3/3] d1-3B 모델 내려받기
python -c "from huggingface_hub import snapshot_download; snapshot_download('LiquidAI/d1-3B')" || goto :fail
python -c "import torch; print('GPU:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
echo.
echo  설치 끝. 이제 start.bat 을 켜면 판단 기관도 따로 창으로 같이 켜진다.
echo  먼저 시험해 보려면: start_decision.bat 을 켜고, 모델이 올라간 뒤 decision_test.bat 을 켜면 된다.
pause
exit /b 0
:fail
echo.
echo  설치 중에 멈췄다. 위의 오류 메시지를 Claude에게 보여줘.
pause
exit /b 1
