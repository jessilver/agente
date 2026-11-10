# AGENTE - Sistema de IA Local (CPU e AMD ROCm)

Este repositório contém a estrutura para execução e testes de modelos de Inteligência Artificial localmente, com suporte a execução otimizada via **CPU** e aceleração por **GPU AMD (ROCm)**.

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
├── data/                    # Pasta para dados e datasets
    └── leis/                # Acervo de legislação (.md / .txt: CF88, Lei 8.080, TO)
├── models/                  # Arquivos e pesos dos modelos
├── training/                # Scripts e logs de treinamento
├── .gitignore
├── README.md                # Este manual de instruções
├── indexador_rag.py         # Motor de indexação paralela com cache incremental em SQLite
├── interface.py             # Interface CustomTkinter (GUI + Motor de IA + Comitê Multi-Agente)
├── rag_busca.py             # Busca nas leis: palavra-chave + vetorial, limpeza, reranker, checagem de citações
├── snapshots.py             # Registro de cada execução em snapshots/ (JSON + Markdown)
├── tests/                   # Testes automáticos (pytest), sem precisar da base
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

## 🧪 Testando os Modelos

* **Teste CPU (Execução Padrão / Validação do Modelo):**
  ```bash
  python3 testModelo.py
  ```
* **Teste de Detecção de GPU (PyTorch):**
  ```bash
  python3 -c "import torch; print('>>> GPU Detectada pelo PyTorch:', torch.cuda.is_assets = torch.cuda.is_available()); print('>>> Versão PyTorch:', torch.__version__)"

```

---

## 🔎 Busca nas Leis (RAG)

A busca fica em `rag_busca.py` e é usada pelo `testModelo.py` e pela `interface.py`. A base `chroma_db_sus/` (≈ 13 GB, fora do git) é **só lida**: nada é gravado nela.

Etapas de `buscar_trechos()`:

1. **Busca vetorial** nas normas constitucionais/federais e na base inteira.
2. **Busca por palavra-chave** para os números de lei citados na consulta (`Lei 8080` → `8.080`) e para `palavras_chave=[...]`.
3. **Descarte** de pastas de trabalho indexadas junto com as leis (`_fonte`, `auditorias`, `backup`, `candidatos`, `_indice`, `_recuperacao`), **limpeza de HTML** e **remoção de duplicados**.
4. **Metadados corrigidos na leitura**: UF pelo caminho (`estaduais/AC/...`) e número pelo cabeçalho `Norma:`.
5. **Reranker** multilíngue (`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, ≈ 470 MB, baixado na primeira execução) ordena os candidatos; trechos abaixo de `NOTA_MINIMA` não vão para o prompt.

Depois da geração, `verificar_citacoes()` compara os artigos e leis citados na resposta com os trechos recuperados, e `anotar_citacoes()` acrescenta um aviso com o que **não** veio da base.

Ajustes comuns (constantes no topo de `rag_busca.py`):

| Constante | Padrão | Efeito |
|---|---|---|
| `NOTA_MINIMA` | `-5.0` | Mais alta = menos trechos, mais relevantes; `None` desliga o corte |
| `N_CANDIDATOS_RERANK` | `50` | Candidatos por busca enviados ao reranker (mais = melhor recall, mais lento) |
| `ESFERAS_PRIORITARIAS` | `constitucional`, `federal` | Esferas buscadas em separado |

## 📸 Snapshots

Cada execução do `testModelo.py` e cada análise da `interface.py` grava `snapshots/<data>_<script>.json` e `.md` com branch/commit, hardware, tempos por etapa, VRAM (em GPU), consulta, trechos (nota, UF, origem), citações confirmadas/fora dos trechos e a resposta. A pasta está no `.gitignore`.

## ✅ Testes Automáticos

Os testes usam uma base falsa, então rodam em segundos e não precisam da `chroma_db_sus/`:

```bash
pip install pytest
python -m pytest tests -q
```

## 📚 Normas Federais de Saúde

A base original não tinha a Lei 8.080/1990, a Lei 8.142/1990, a LC 141/2012 nem o Decreto 7.508/2011. Elas ficam em `data/normas_federais_saude/` (texto oficial do Planalto, sem os trechos revogados, um bloco por artigo) e são geradas e indexadas assim:

```bash
python scripts/importar_normas_planalto.py
python indexador_rag.py --base data/normas_federais_saude
```

O indexador só adiciona os arquivos dessa pasta (caminhos `federais/saude/...`); o restante da base não é tocado. Para acrescentar outra norma, inclua-a na lista `NORMAS` do script.

> Antes de qualquer escrita na `chroma_db_sus/`, faça um backup, por exemplo:
> `7z a -tzip -mx=1 ../backups/chroma_db_sus_AAAA-MM-DD.zip ./chroma_db_sus` e teste com `7z t`.

## ⚠️ Limitações Conhecidas e Próximos Passos

Pendências identificadas e deixadas para as próximas etapas, da mais impactante para a menos:

| # | Limitação | Impacto | Por onde começar |
|---|---|---|---|
| 1 | A base usa embeddings em inglês (`all-MiniLM-L6-v2`) | Em perguntas sem termos exatos, os trechos certos muitas vezes nem chegam aos candidatos; o reranker não consegue compensar | Reindexar com um modelo multilíngue (ex.: `intfloat/multilingual-e5-base`) numa **base nova**, comparar com os snapshots e só então substituir a compartilhada |
| 2 | O indexador inclui pastas de trabalho (`_fonte`, `auditorias`, `backup`, `candidatos`, `_indice`) e marca toda norma de `estaduais/` como UF `TO` | Duplicados e metadados errados; hoje corrigidos só na leitura (`rag_busca.py`) | `coletar_arquivos()` e `classificar_documento()` em `indexador_rag.py`; vale para a próxima reindexação |
| 3 | `NOTA_MINIMA` separa pouco | Trechos sem relação ainda passam com notas entre -2 e -5 | Testar um reranker maior (ex.: `BAAI/bge-reranker-v2-m3`, viável só em GPU) e recalibrar a nota com os snapshots |
| 4 | A checagem de citações compara números, não o conteúdo | Uma citação com o número certo e o texto errado não é marcada | Comparar o trecho entre aspas da resposta com o texto recuperado |
| 5 | A linha NVIDIA (`nvidia/develop`) ainda não tem reranker, snapshots nem as correções de leitura | `NVIDIA/testModeloNvidia.py` e `NVIDIA/interfaceNvidia.py` usam a busca anterior | `sync/develop-to-nvidia` + branch `feat/nvidia-*`, conforme o `BRANCHING.md` |
| 6 | `data/leis` é um gitlink sem `.gitmodules` | A pasta vem vazia em qualquer clone | O autor do commit `c6683e1` deve adicionar o `.gitmodules` ou versionar os arquivos |
