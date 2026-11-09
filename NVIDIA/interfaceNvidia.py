# ==========================================
# Interface gráfica do Agente rodando em placas NVIDIA (CUDA)
# Reaproveita a interface.py da raiz e só troca o carregamento dos modelos para a GPU.
# ==========================================

import os
import sys

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

diretorio_raiz = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, diretorio_raiz)

import chromadb  # noqa: E402
import interface  # noqa: E402


class _TokenizerNaGPU:
    """Envia os tensores gerados pelo tokenizer direto para a GPU, sem alterar o MotorIA original."""

    def __init__(self, tokenizer, dispositivo):
        self._tokenizer = tokenizer
        self._dispositivo = dispositivo

    def __call__(self, *args, **kwargs):
        return self._tokenizer(*args, **kwargs).to(self._dispositivo)

    def __getattr__(self, nome):
        return getattr(self._tokenizer, nome)


class MotorIANvidia(interface.MotorIA):
    def carregar_motores(self):
        """Carrega o banco de leis e o modelo LLaMA na VRAM da placa NVIDIA."""
        if not torch.cuda.is_available():
            raise RuntimeError("Nenhuma GPU CUDA detectada pelo PyTorch. Verifique o driver NVIDIA ('nvidia-smi').")

        dispositivo = "cuda"
        tipo_dado = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        caminho_chroma = os.path.join(diretorio_raiz, "chroma_db_sus")
        modelo_base_id = "meta-llama/Llama-3.2-3B-Instruct"

        client_chroma = chromadb.PersistentClient(path=caminho_chroma)
        self.collection = client_chroma.get_or_create_collection(name="base_juridica_sus")

        self.encoder_rag = SentenceTransformer("all-MiniLM-L6-v2", device=dispositivo)

        tokenizer = AutoTokenizer.from_pretrained(modelo_base_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = _TokenizerNaGPU(tokenizer, dispositivo)

        self.model = AutoModelForCausalLM.from_pretrained(
            modelo_base_id,
            dtype=tipo_dado,
            device_map={"": 0},
            attn_implementation="sdpa",
        )
        self.model.eval()
        self.pronto = True


class InterfaceGraficaNvidia(interface.InterfaceGrafica):
    def __init__(self):
        super().__init__()
        self.title("Agente de Auditoria de Conformidade do SUS (NVIDIA/CUDA)")
        self.ia = MotorIANvidia()
        self.btn_carregar.configure(text="1. Ligar Inteligência Artificial (Carregar VRAM)")

    def evento_carregar_ia(self):
        super().evento_carregar_ia()
        self.status_label.configure(text="Status: Carregando modelo e banco de leis na VRAM... (Aguarde)")


if __name__ == "__main__":
    app = InterfaceGraficaNvidia()
    app.mainloop()
