@echo off
setlocal

echo [1/3] Criando ambiente virtual para NVIDIA (.venv_nvidia)...
py -3.12 -m venv .venv_nvidia

if not exist ".venv_nvidia\Scripts\activate.bat" (
    echo ERRO: ambiente virtual nao foi criado corretamente.
    exit /b 1
)

echo [2/3] Ativando o ambiente virtual...
call .venv_nvidia\Scripts\activate.bat

python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt --no-deps

echo [3/3] Instalacao concluida.
echo Ambiente prontos para uso com GPU NVIDIA.
echo Execute: python NVIDIA\indexNvidia.py
pause
