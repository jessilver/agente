# AGENTE - Sistema de IA Local (CPU, AMD ROCm e NVIDIA CUDA)

Este repositório contém a estrutura para execução e testes de modelos de Inteligência Artificial localmente, com suporte a execução otimizada via **CPU** e aceleração por **GPU AMD (ROCm)** ou **GPU NVIDIA (CUDA)**.

O projeto faz o *fine-tuning* com **LoRA/QLoRA** do modelo `meta-llama/Llama-3.2-3B` sobre um dataset de perguntas e respostas da área da saúde (`data/dataset_mestre.jsonl`) e valida o resultado com scripts de teste.

> Fork de [gabrielMartiliano/agente](https://github.com/gabrielMartiliano/agente).

---

## 📂 Estrutura do Projeto

```text
contract-review-agent/
├── AMD/
│   ├── indexAmd.py                       # Treinamento LoRA (4-bit) para AMD ROCm
│   └── Ubuntu-instalar_ambienteAMD-.sh   # Instalação do ambiente AMD (Ubuntu)
├── CPU/
│   ├── indexCPU.py                       # Treinamento LoRA em CPU
│   └── win_ambiente_cpu.bat              # Instalação do ambiente CPU (Windows)
├── NVIDIA/
│   ├── indexNvidia.py                    # Treinamento QLoRA (4-bit) para NVIDIA CUDA
│   ├── testModeloNvidia.py               # Teste com RAG na GPU (versão CUDA do testModelo.py)
│   ├── interfaceNvidia.py                # Interface gráfica com os modelos na VRAM
│   ├── README.md                         # Requisitos e execução na NVIDIA
│   └── Fedora-instalar_ambienteNVIDIA.sh # Instalação do ambiente NVIDIA (Fedora)
├── data/
│   ├── leis/                             # Acervo de legislação (.md / .txt: CF88, Lei 8.080, TO)
│   └── dataset_mestre.jsonl              # Dataset de treino (um JSON por linha, campo "text")
├── models/                               # (gerado) adaptador LoRA final em lora_model_saude/
├── training/                             # (gerado) checkpoints em checkpoints_salvos/
├── chroma_db_sus/                        # (gerado/baixado, fora do git) base vetorial do RAG
├── .venv_linux/                          # (gerado) ambiente virtual Python (Linux)
├── .gitignore
├── README.md                             # Este manual de instruções
├── indexador_rag.py                      # Indexação das leis no ChromaDB com cache incremental em SQLite
├── interface.py                          # Interface CustomTkinter (GUI + RAG) em CPU
├── prompt.txt                            # Documento de exemplo analisado pelo testModelo.py
├── verificardb.py                        # Inspeção da base vetorial
├── requirements.txt                      # Dependências do projeto
├── testModelo.py                         # Teste com RAG em CPU
└── testModeloGPU.py                      # Teste do modelo treinado em GPU
```

Os itens marcados como *(gerado)* são criados durante a instalação/treino e não são versionados (ver `.gitignore`).

---

## ⚙️ Requisitos do Sistema

* **Sistema Operacional:** Linux (Ubuntu 22.04 / 24.04 recomendado para AMD ROCm, Fedora para NVIDIA CUDA) ou Windows (para execução em CPU).
* **Python:** Versão **3.10 ou 3.11** (recomendado 3.11, para compatibilidade com PyTorch). Nos scripts de instalação NVIDIA e AMD o Python 3.11 é fixado via [`uv`](https://docs.astral.sh/uv/) dentro do `.venv_linux`, sem alterar o Python do sistema (o Fedora 44, por exemplo, traz o 3.14).
* **Hardware para GPU AMD:** Placa de vídeo AMD Radeon compatível com ROCm (ex: arquitetura Navi / RDNA otimizada com variáveis de ambiente específicas).
* **Hardware para GPU NVIDIA:** Placa de vídeo NVIDIA compatível com CUDA 12.1 (wheels `cu121` do PyTorch), com driver proprietário instalado (via RPM Fusion no Fedora).
* **Memória:**
  * Treino em GPU (4-bit): cabe em **6 GB de VRAM** (validado numa RTX 4050 Laptop).
  * Teste em GPU (`testModeloGPU.py`, bf16): precisa de **mais de 6 GB** de VRAM, pois os pesos ocupam ≈ 6,4 GB (ver [limitações](#-solução-de-problemas-e-limitações-conhecidas)).
  * Treino em CPU (float32): ≈ 13 GB de RAM só para os pesos (3,2 B de parâmetros × 4 bytes).
* **Disco:** ≈ 5 GB para o ambiente Python/PyTorch-CUDA + ≈ 6 GB para os pesos do modelo no cache do Hugging Face (`~/.cache/huggingface`).
* **Conta no Hugging Face** com acesso ao modelo `meta-llama/Llama-3.2-3B` (modelo "gated").

### 🖥️ Ambiente de desenvolvimento validado

Fluxo NVIDIA/CUDA executado de ponta a ponta (instalação → treino) em 18/09/2026:

| Item | Versão |
|---|---|
| Sistema operacional | Fedora Linux 44 (Workstation), kernel 7.2.5 |
| GPU / driver | NVIDIA GeForce RTX 4050 Laptop (6 GB), driver 615.71.09 (RPM Fusion, `akmod-nvidia`) |
| Python / gerenciador | 3.11.16 / `uv` 0.12.17 |
| PyTorch | 2.5.1+cu121 |
| transformers / trl / peft | 5.17.0 / 1.13.0 / 0.21.0 |
| bitsandbytes / accelerate / datasets | 0.50.2 / 1.15.0 / 5.0.1 |

> O `requirements.txt` define apenas versões mínimas (`>=`); a tabela registra as versões resolvidas nessa validação.

---

## 🚀 Guia de Instalação e Execução

### 1. Clonar o repositório (fork)

```bash
git clone https://github.com/ilwca/contract-review-agent.git
cd contract-review-agent

# (opcional) acompanhar o repositório original
git remote add upstream https://github.com/gabrielMartiliano/agente.git
```

### 2. Criar o ambiente

| Hardware | Como criar o ambiente |
|---|---|
| NVIDIA (Fedora) | `bash NVIDIA/Fedora-instalar_ambienteNVIDIA.sh` — ver [seção NVIDIA](#-executando-em-ambiente-nvidia-cuda--fedora) |
| AMD (Ubuntu) | `bash AMD/Ubuntu-instalar_ambienteAMD-.sh` — ver [seção AMD](#-executando-em-ambiente-amd-rocm) |
| CPU (Linux/Windows) | Manual ou `CPU\win_ambiente_cpu.bat` — ver [seção CPU](#-executando-em-ambiente-cpu) |

### 3. Autenticar no Hugging Face (todos os hardwares)

O modelo base `meta-llama/Llama-3.2-3B` é "gated". Aceite a licença em https://huggingface.co/meta-llama/Llama-3.2-3B e autentique localmente **antes de treinar**:

```bash
hf auth login
```

> O comando antigo `huggingface-cli login` foi descontinuado e não funciona mais nas versões atuais do `huggingface_hub`.

### 4. Treinar e testar

Siga a seção do seu hardware abaixo. O fluxo é sempre **treinar** (`index*.py`, gera o adaptador em `models/lora_model_saude/`) e depois **testar** (`testModelo*.py`, que depende desse adaptador).

---

## 🔌 Executando em Ambiente NVIDIA (CUDA) — Fedora

### Instalação do ambiente

```bash
# Dê permissão de execução e rode o script de instalação do ambiente
chmod +x NVIDIA/Fedora-instalar_ambienteNVIDIA.sh
bash NVIDIA/Fedora-instalar_ambienteNVIDIA.sh
```

O script executa, em ordem:

1. Instala as dependências de build (`gcc`, `gcc-c++`, `make`, `git`) via `dnf`, se faltarem.
2. Confere o driver com `nvidia-smi`. Se não estiver instalado, habilita o RPM Fusion e instala o `akmod-nvidia` — nesse caso é necessário **reiniciar o sistema** e rodar o script novamente.
3. Instala o `uv` e o Python 3.11 e cria o ambiente virtual isolado `.venv_linux`.
4. Instala o PyTorch com CUDA 12.1 (índice `cu121`), depois as dependências do `requirements.txt` e o `huggingface_hub`.

> **Rodar novamente:** se o `.venv_linux` já existir, o script aborta com `A virtual environment already exists`. Para reinstalar do zero, remova o ambiente antes: `rm -rf .venv_linux`.

### Treinar e testar

```bash
# Ative o ambiente virtual
source .venv_linux/bin/activate

# Confirme que o PyTorch está enxergando a GPU
python3 -c "import torch; print('CUDA disponível:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'sem GPU')"

# Autentique no Hugging Face (uma única vez, veja o passo 3 do guia)
hf auth login

# Execute o treinamento QLoRA otimizado para NVIDIA/CUDA
python3 NVIDIA/indexNvidia.py

# Execute o script de testes na GPU (atenção à limitação de 6 GB de VRAM abaixo)
python3 testModeloGPU.py
```

**Saída esperada do treino** (validada na RTX 4050): a *loss* começa em ≈ 2,0 e cai para ≈ 0,2 ao longo dos 60 passos; o adaptador (≈ 18 MB) e o tokenizer são salvos em `models/lora_model_saude/` e os checkpoints em `training/checkpoints_salvos/`. A primeira execução baixa ≈ 6 GB do modelo base.

---

## 🔌 Executando em Ambiente AMD (ROCm)

Para rodar scripts utilizando a aceleração da GPU AMD no Linux, é necessário configurar as variáveis de ambiente corretas para evitar erros de compatibilidade de arquitetura (especialmente em placas Radeon suportadas via override).

Instalação do ambiente (o script assume que o `uv` já está instalado):
```bash
bash AMD/Ubuntu-instalar_ambienteAMD-.sh
```

Treino e teste:
```bash
# Ative o ambiente virtual
source .venv_linux/bin/activate

# Defina as variáveis de ambiente para o ROCm (exemplo para arquiteturas RDNA/GFX10.3)
export HSA_ENABLE_SDMA=0
export HSA_OVERRIDE_GFX_VERSION=10.3.0

# Execute o treinamento LoRA otimizado para AMD
python3 AMD/indexAmd.py

# Execute o script de testes na GPU
python3 testModeloGPU.py
```

---

## 💻 Executando em Ambiente CPU

### No Linux:
```bash
python3 -m venv .venv_linux
source .venv_linux/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

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
(No PowerShell, ative o ambiente com `.venv\Scripts\Activate.ps1`.)

---

## 📄 Formato do Dataset

`data/dataset_mestre.jsonl` contém um objeto JSON por linha, com o campo `text` no formato pergunta/resposta (o mesmo formato usado nos prompts de teste):

```json
{"text": "### Pergunta: O que é saúde pública?\n### Resposta: Saúde pública é a área da ciência voltada para a prevenção de doenças e promoção da qualidade de vida na população."}
```

---

## 🧠 Configuração de Treino

Comum aos três scripts: modelo `meta-llama/Llama-3.2-3B`, LoRA com `r=16`, `lora_alpha=16`, `dropout=0.05` nos módulos `q_proj`, `k_proj`, `v_proj` e `o_proj`, *batch* 1 com acumulação de gradiente 4 e taxa de aprendizado `2e-4`.

| Parâmetro | CPU | AMD (ROCm) | NVIDIA (CUDA) |
|---|---|---|---|
| Precisão / quantização | float32, sem quantização | 4-bit NF4 | 4-bit NF4 + *double quant* |
| Passos (`max_steps`) | 10 | 60 | 60 |
| Otimizador | `adamw_torch` | `adamw_8bit` | `adamw_8bit` |
| Tokenizer salvo com o adaptador | não | não | sim |

Nos scripts de GPU o tipo de cálculo (*compute dtype*) é bf16 quando suportado, ou fp16 caso contrário.

---

## 🧪 Testando os Modelos

* **Teste CPU (Execução Padrão / Validação do Modelo):**
  ```bash
  python3 testModelo.py
  ```
* **Teste em GPU (AMD ou NVIDIA):**
  ```bash
  python3 testModeloGPU.py
  ```
* **Teste de Detecção de GPU (PyTorch):**
  ```bash
  python3 -c "import torch; print('>>> GPU Detectada pelo PyTorch:', torch.cuda.is_available()); print('>>> Versão PyTorch:', torch.__version__)"
  ```

---

## 🩺 Solução de Problemas e Limitações Conhecidas

* **`nvidia-smi` falha ou `CUDA disponível: False` (NVIDIA/Fedora):** confirme que o módulo do driver existe para o kernel em uso com `rpm -q kmod-nvidia-$(uname -r)`. Após atualizar o kernel, aguarde o `akmods` recompilar o módulo e reinicie. Com **Secure Boot** ativo, o módulo precisa estar assinado (veja o guia de Secure Boot do RPM Fusion). Se o driver estiver ok e o PyTorch ainda não enxergar a GPU, confira `python3 -c "import torch; print(torch.version.cuda)"` (esperado: `12.1`).
* **Erro 401/403 ao baixar o modelo:** a licença do modelo não foi aceita ou você não está autenticado — veja o passo 3 do guia (`hf auth login`).
* **`A virtual environment already exists` ao reexecutar o script NVIDIA:** remova o ambiente antigo com `rm -rf .venv_linux` e rode o script novamente.
* **`testModeloGPU.py` esgota a memória em GPUs de 6 GB (ex.: RTX 4050 Laptop):** o script carrega o modelo base em bf16 (≈ 6,4 GB), mais do que a VRAM disponível. O treino em 4-bit funciona normalmente. Alternativas: usar uma GPU com mais VRAM, rodar `python3 testModelo.py` (CPU, mais lento) ou carregar o modelo em 4-bit (`BitsAndBytesConfig`) no teste, como é feito no treino.
* **Dataset pequeno e *overfitting*:** o `dataset_mestre.jsonl` de exemplo tem apenas 2 registros. Com 60 passos o modelo percorre o dataset 60 vezes (a *loss* estabiliza em ≈ 0,2 a partir do passo 15) e memoriza os exemplos, sem conjunto de validação para medir generalização. Os prompts de teste incluem uma pergunta idêntica e uma paráfrase do treino, então também não medem generalização. Para uso real, amplie o dataset, separe uma parte para validação e ajuste `max_steps`.
