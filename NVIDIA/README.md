# NVIDIA (CUDA)

Scripts específicos para placas NVIDIA. Rode sempre a partir da raiz do projeto.

| Script | O que faz |
|---|---|
| `indexNvidia.py` | Treinamento QLoRA (4-bit) do `Llama-3.2-3B` |
| `testModeloNvidia.py` | Versão GPU do `testModelo.py`: RAG (ChromaDB) + `Llama-3.2-3B-Instruct` analisando o `prompt.txt` |
| `interfaceNvidia.py` | Versão GPU da `interface.py`: mesma interface gráfica, com os modelos carregados na VRAM |

## Requisitos

* PyTorch com CUDA (`torch.cuda.is_available()` deve retornar `True`).
* `chromadb`, `sentence-transformers` e `customtkinter` instalados.
* Acesso ao modelo "gated" `meta-llama/Llama-3.2-3B-Instruct` no Hugging Face.
* **VRAM:** ≈ 6,1 GB alocados em bf16 (validado numa RTX 4070 Laptop, 8 GB).
* `testModeloNvidia.py` e `interfaceNvidia.py` dependem do `prompt.txt` e da `interface.py` da raiz, que chegam nesta linha pelo `sync/develop-to-nvidia`.

## Execução

```bash
python NVIDIA/testModeloNvidia.py
python NVIDIA/interfaceNvidia.py
```

Se a base `chroma_db_sus/` não existir, ela é criada vazia e o modelo responde sem trechos de lei recuperados. Para popular, rode `python indexador_rag.py` com as leis em `data/leis/`.
