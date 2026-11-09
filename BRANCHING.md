# Modelo de Branches

O projeto mantém **duas linhas paralelas**, uma por família de GPU. Cada linha tem uma branch estável e uma de integração.

## 🗂️ Ambiente × Branch

| Ambiente | Estável | Integração | Pastas próprias |
|---|---|---|---|
| AMD (ROCm) + CPU | `main` | `develop` | `AMD/`, `CPU/` |
| NVIDIA (CUDA) | `nvidia/main` | `nvidia/develop` | `NVIDIA/` |

Arquivos **compartilhados** pelas duas linhas: `data/`, `requirements.txt`, `testModelo.py`, `testModeloGPU.py`, `README.md` e `.gitignore`.

## 🔀 Fluxo de PR

```text
feat/dataset-*, fix/*, feat/amd-*, feat/cpu-*  ──PR──▶ develop ──PR──▶ main
                                                     │
                                                     └──merge (PR sync/develop-to-nvidia)──▶ nvidia/develop ──PR──▶ nvidia/main
feat/nvidia-*  ──────────────────────────────────────────────────────PR──▶ nvidia/develop
```

1. **Datasets e correções comuns** (qualquer coisa em arquivos compartilhados) entram **sempre por `develop`**.
2. Depois, `develop` é propagada para `nvidia/develop` por **merge** (nunca cherry-pick), via PR de uma branch `sync/develop-to-nvidia`. Assim o histórico fica rastreável e não há commits duplicados.
3. Código específico de NVIDIA (`NVIDIA/`) entra **só** por `nvidia/develop`.
4. Releases: PR `develop → main` e PR `nvidia/develop → nvidia/main`.
5. **Nunca** fazer merge de `nvidia/*` em `develop` ou `main`.
6. `main` e `nvidia/main` não aceitam push direto: toda mudança chega por PR revisado.

## 🏷️ Convenção de nomes

| Prefixo | Uso | Base do PR |
|---|---|---|
| `feat/dataset-<nn>-<tema>` | Novo bloco de dataset em `data/` | `develop` |
| `feat/amd-<desc>` | Mudanças em `AMD/` | `develop` |
| `feat/cpu-<desc>` | Mudanças em `CPU/` | `develop` |
| `feat/nvidia-<desc>` | Mudanças em `NVIDIA/` | `nvidia/develop` |
| `fix/<desc>` | Correção em código ou dados compartilhados | `develop` |
| `docs/<desc>` | Documentação | `develop` (ou `nvidia/develop`, se for só NVIDIA) |
| `sync/develop-to-nvidia` | Propagação `develop → nvidia/develop` | `nvidia/develop` |

> Em forks, abra o PR a partir de uma branch de feature, não do `main` do fork. Assim o `main` do fork continua livre para acompanhar o upstream.

## 📝 README e documentação por ambiente

Para evitar conflitos nos merges `develop → nvidia/develop`:

- `README.md` (raiz): só o conteúdo comum, com uma tabela que aponta para a documentação de cada ambiente.
- Detalhes de instalação e execução ficam em `AMD/README.md`, `CPU/README.md` e `NVIDIA/README.md`. Cada linha só edita o arquivo do seu ambiente.

## 💾 Backup e arquivamento

- Antes de reorganizar branches, crie tags `backup/<branch>-AAAA-MM-DD` apontando para a ponta de cada branch remota.
- Branches antigas só são removidas **depois** que todos os PRs que as usam como base ou head forem fechados. Antes de remover, crie a tag `archive/<branch>`.
- Nunca use force-push, rebase ou delete em branches que são base ou head de PR aberto.
