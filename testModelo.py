import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import chromadb
from sentence_transformers import SentenceTransformer
from rag_busca import buscar_trechos, formatar_contexto

print("=== TESTANDO O AGENTE COM RAG (EXCLUSIVO PARA CPU) ===")

# ==========================================
# 1. CAMINHOS E BANCO VETORIAL
# ==========================================
diretorio_script = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(diretorio_script) in ["cpu", "amd", "nvidia"]:
    caminho_txt = os.path.join(diretorio_script, "..", "prompt.txt")
    caminho_chroma = os.path.join(diretorio_script, "..", "chroma_db_sus")
else:
    caminho_txt = os.path.join(diretorio_script, "prompt.txt")
    caminho_chroma = os.path.join(diretorio_script, "chroma_db_sus")

modelo_base_id = "meta-llama/Llama-3.2-3B-Instruct"

print(f"Conectando à base vetorial RAG em: {caminho_chroma}...")
client_chroma = chromadb.PersistentClient(path=caminho_chroma)
collection = client_chroma.get_or_create_collection(name="base_juridica_sus")

print("Carregando modelo de embedding leve (SentenceTransformer em CPU)...")
encoder_rag = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# ==========================================
# 2. CARREGAMENTO DO MODELO E TOKENIZER
# ==========================================
print(f"\nCarregando tokenizer de {modelo_base_id}...")
tokenizer = AutoTokenizer.from_pretrained(modelo_base_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("\nCarregando modelo base na memória RAM...")
model = AutoModelForCausalLM.from_pretrained(
    modelo_base_id,
    dtype=torch.float32,
    device_map="cpu",
    low_cpu_mem_usage=True
)
model.eval()
print("Modelo pronto para uso!\n")

# ==========================================
# 3. LEITURA E CONSULTA RAG FOCADA
# ==========================================
if not os.path.exists(caminho_txt):
    raise FileNotFoundError(f"Arquivo não encontrado: {caminho_txt}")

with open(caminho_txt, "r", encoding="utf-8") as f:
    texto_entrada = f.read().strip()

# Termo focado exclusivamente nos princípios violados para a busca vetorial
termo_busca = "gratuidade do SUS taxa cobranca universalidade igualdade acesso atendimento saude Lei 8080 artigo 7 196"

print(f"Buscando leis relevantes no ChromaDB para: '{termo_busca}'")
trechos = buscar_trechos(collection, encoder_rag, termo_busca, n_final=3)
contexto_legal_recuperado = formatar_contexto(trechos)

# Estruturação usando o formato oficial de mensagens do LLaMA-3.2-Instruct
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
    {"role": "user", "content": user_content}
]

prompt_formatado = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

print("\n" + "=" * 60)
print("PROMPT FINAL ENVIADO AO MODELO:")
print("-" * 60)
print(prompt_formatado)
print("=" * 60 + "\n")

# ==========================================
# 4. GERAÇÃO DETERMINÍSTICA E LIMPA NA CPU
# ==========================================
inputs = tokenizer(prompt_formatado, return_tensors="pt")

print("Gerando parecer de auditoria...")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=400,
        do_sample=False,
        repetition_penalty=1.15,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id
    )

# Decodificação isolando apenas a resposta gerada
resposta_completa = tokenizer.decode(outputs[0], skip_special_tokens=True)
resposta_gerada = resposta_completa.split("assistant")[-1].strip()

print("\n" + "=" * 60)
print("RESPOSTA DO MODELO:")
print("-" * 60)
print(resposta_gerada)
print("=" * 60)