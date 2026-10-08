# AGENTE - Sistema de IA Local (CPU e AMD ROCm)

Este repositório contém a estrutura para execução e testes de modelos de Inteligência Artificial localmente, com suporte a execução otimizada via **CPU**, **GPU NVIDIA (CUDA)** e **GPU AMD (ROCm)**.

---

## 📂 Estrutura do Projeto

```text
AGENTE/
├── 
├── AMD/
│   ├── indexAmd.py          # Script principal/configuração para AMD ROCm
│   └── Ubuntu-instalar_ambienteAMD-.sh # Script de automação para instalação do ROCm
├── CPU/
│   ├── indexCPU.py          # Script principal/configuração para CPU
│   └── win_ambiente_cpu.bat # Script de automação para Windows (CPU)
├── NVIDIA/
│   ├── indexNvidia.py       # Script principal/configuração para GPU NVIDIA
│   └── win_ambiente_nvidia.bat # Script de automação para Windows (NVIDIA)
├── data/                    # Pasta para dados e datasets
├── models/                  # Arquivos e pesos dos modelos
├── training/                # Scripts e logs de treinamento
├── .gitignore
├── README.md                # Este manual de instruções
├── requirements.txt         # Dependências do projeto
├── testModelo.py            # Script genérico de teste do modelo (CPU/Padrão)
└── testModeloGPU.py         # Script de validação/teste utilizando GPU
```

---

## ⚙️ Requisitos do Sistema

* **Sistema Operacional:** Linux (Ubuntu 22.04 / 24.04 recomendado para AMD ROCm) ou Windows (para execução em CPU).
* **Python:** Versão **3.10 ou 3.11** (recomendado para compatibilidade com PyTorch).
* **Hardware para GPU AMD:** Placa de vídeo AMD Radeon compatível com ROCm (ex: arquitetura Navi / RDNA otimizada com variáveis de ambiente específicas).

---

## 🚀 Guia de Instalação e Execução

### 1. Clonar e Acessar o Repositório
Abra o terminal na pasta raiz do projeto `AGENTE`.

### 2. Configuração do Ambiente Virtual (Linux)

Caso ainda não tenha criado o ambiente virtual:
```bash
python3 -m venv .venv_linux

linux
source .venv_linux/bin/activate
win
.venv\Scripts\Activate.ps1
pip install --upgrade pip
```

### 3. Instalação das Dependências
Com o ambiente ativado, instale os pacotes listados no `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

## 🔌 Executando em Ambiente AMD (ROCm)

Para rodar scripts utilizando a aceleração da GPU AMD no Linux, é necessário configurar as variáveis de ambiente corretas para evitar erros de compatibilidade de arquitetura (especialmente em placas Radeon suportadas via override):

```bash
# Ative o ambiente virtual
source .venv_linux/bin/activate

# Defina as variáveis de ambiente para o ROCm (exemplo para arquiteturas RDNA/GFX10.3)
export HSA_ENABLE_SDMA=0
export HSA_OVERRIDE_GFX_VERSION=10.3.0

# Execute o script de testes na GPU
python3 testModeloGPU.py
```

Se preferir rodar a automação de instalação do ambiente AMD:
```bash
bash AMD/Ubuntu-instalar_ambienteAMD-.sh
```

---

## 💻 Executando em Ambiente CPU

### No Linux:
```bash
source .venv_linux/bin/activate
python3 CPU/indexCPU.py
```

### No Windows:
Basta executar o arquivo de lote automatizado na pasta CPU:
```cmd
CPU\win_ambiente_cpu.bat
```
Ou via prompt de comando:
```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python CPU/indexCPU.py
```

---

## 🎮 Executando em Ambiente NVIDIA (Windows)

Para rodar no Windows com GPU NVIDIA, use a venv específica do projeto:

```powershell
cd D:\Codes\PYTHON\UFT\agente
py -3.12 -m venv .venv_nvidia
.\.venv_nvidia\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python .\NVIDIA\indexNvidia.py
```

Ou use o script pronto:

```powershell
.\NVIDIA\win_ambiente_nvidia.bat
```

> Se o modelo escolhido for restrito da Meta (como `meta-llama/Llama-3.2-3B`), será necessário autenticar-se no Hugging Face com `huggingface-cli login` ou usar um modelo público alternativo.

---

## 🧪 Testando os Modelos

* **Teste CPU (Execução Padrão / Validação do Modelo):**
  ```bash
  python3 testModelo.py
  ```
* **Teste de Detecção de GPU (PyTorch):**
  ```bash
  python3 -c "import torch; print('>>> GPU Detectada pelo PyTorch:', torch.cuda.is_assets = torch.cuda.is_available()); print('>>> Versão PyTorch:', torch.__version__)"