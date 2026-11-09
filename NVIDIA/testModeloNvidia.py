# ==========================================
# Teste do Agente com RAG em placas NVIDIA (CUDA)
# Versão GPU do testModelo.py (que roda exclusivamente em CPU).
# ==========================================

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import sys
import chromadb
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
from rag_busca import buscar_trechos, formatar_contexto  # noqa: E402

print("=== TESTANDO O AGENTE COM RAG (NVIDIA/CUDA) ===")

if not torch.cuda.is_available():
    raise RuntimeError(
        "Nenhuma GPU CUDA detectada pelo PyTorch. Verifique o driver NVIDIA ('nvidia-smi') "
        "e se o PyTorch foi instalado com suporte a CUDA (índice https://download.pytorch.org/whl/cu121)."
    )

dispositivo = "cuda"
tipo_dado = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def mostrar_memoria():
    mem_alocada = torch.cuda.memory_allocated() / (1024 ** 3)
    mem_reservada = torch.cuda.memory_reserved() / (1024 ** 3)
    print(f"[{torch.cuda.get_device_name(0)}] VRAM Alocada: {mem_alocada:.2f} GB | Reservada: {mem_reservada:.2f} GB")


# ==========================================
# 1. CAMINHOS E BANCO VETORIAL
# ==========================================
diretorio_raiz = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
caminho_txt = os.path.join(diretorio_raiz, "prompt.txt")
caminho_chroma = os.path.join(diretorio_raiz, "chroma_db_sus")

modelo_base_id = "meta-llama/Llama-3.2-3B-Instruct"

print(f"Conectando à base vetorial RAG em: {caminho_chroma}...")
client_chroma = chromadb.PersistentClient(path=caminho_chroma)
collection = client_chroma.get_or_create_collection(name="base_juridica_sus")

print("Carregando modelo de embedding (SentenceTransformer na GPU)...")
encoder_rag = SentenceTransformer("all-MiniLM-L6-v2", device=dispositivo)

# ==========================================
# 2. CARREGAMENTO DO MODELO E TOKENIZER
# ==========================================
print(f"\nCarregando tokenizer de {modelo_base_id}...")
tokenizer = AutoTokenizer.from_pretrained(modelo_base_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"\nCarregando modelo base na VRAM ({tipo_dado})...")
model = AutoModelForCausalLM.from_pretrained(
    modelo_base_id,
    dtype=tipo_dado,
    device_map={"": 0},
    attn_implementation="sdpa",
)
model.eval()
mostrar_memoria()
print("Modelo pronto para uso!\n")

# ==========================================
# 3. LEITURA E CONSULTA RAG FOCADA
# ==========================================
if not os.path.exists(caminho_txt):
    raise FileNotFoundError(f"Arquivo não encontrado: {caminho_txt}")

with open(caminho_txt, "r", encoding="utf-8") as f:
    texto_entrada = f.read().strip()

termo_busca = "gratuidade do SUS taxa cobranca universalidade igualdade acesso atendimento saude Lei 8080 artigo 7 196"

print(f"Buscando leis relevantes no ChromaDB para: '{termo_busca}'")
trechos = buscar_trechos(collection, encoder_rag, termo_busca, n_final=3)
contexto_legal_recuperado = formatar_contexto(trechos)

system_prompt = (
    "Você é um auditor sênior de conformidade do Sistema Único de Saúde (SUS). "
    "Sua função é emitir um parecer técnico estritamente fundamentado nas leis apresentadas. "
    "Utilize os vereditos padronizados: 'Conforme' ou 'Inconforme'."
)

user_content = (
    f"[CONTEXTO JURÍDICO RECUPERADO DA BASE DE LEIS (SUS / TO)]\n"
    f"{contexto_legal_recuperado}\n\n"
    f"[DOCUMENTO SOB ANÁLISE E DIRETRIZES]\n"
    f"{texto_entrada}"
)

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_content},
]

prompt_formatado = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

print("\n" + "=" * 60)
print("PROMPT FINAL ENVIADO AO MODELO:")
print("-" * 60)
print(prompt_formatado)
print("=" * 60 + "\n")

# ==========================================
# 4. GERAÇÃO DETERMINÍSTICA NA GPU
# ==========================================
inputs = tokenizer(prompt_formatado, return_tensors="pt").to(dispositivo)

print("Gerando parecer de auditoria...")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=400,
        do_sample=False,
        repetition_penalty=1.15,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
    )
mostrar_memoria()

resposta_gerada = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

print("\n" + "=" * 60)
print("RESPOSTA DO MODELO:")
print("-" * 60)
print(resposta_gerada)
print("=" * 60)
