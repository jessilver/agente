import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

print("=== TESTANDO O AGENTE (EXCLUSIVO PARA GPU) ===")

# Verifica se a GPU está realmente acessível antes de continuar
if not torch.cuda.is_available():
    raise RuntimeError("Nenhuma GPU detectada pelo PyTorch. Verifique a instalação do CUDA/ROCm.")

# ==========================================
# 1. AJUSTE DE CAMINHOS RELATIVOS
# ==========================================
diretorio_script = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(diretorio_script) in ["cpu", "amd", "nvidia"]:
    caminho_modelo_final = os.path.join(diretorio_script, "..", "models", "lora_model_saude")
else:
    caminho_modelo_final = os.path.join(diretorio_script, "models", "lora_model_saude")

modelo_base_id = "meta-llama/Llama-3.2-3B"

# ==========================================
# 2. CARREGAMENTO DOS MODELOS E OTIMIZAÇÃO
# ==========================================
modelo_base_id = os.environ.get("MODEL_ID", modelo_base_id)
modelo_fallback = os.environ.get("FALLBACK_MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

try:
    print(f"Carregando tokenizer do {modelo_base_id}...")
    tokenizer = AutoTokenizer.from_pretrained(modelo_base_id, token=True)

    print("\nCarregando modelo base na memória VRAM da Placa de Vídeo...")
    model = AutoModelForCausalLM.from_pretrained(
        modelo_base_id,
        dtype=torch.bfloat16,
        device_map={"": 0},
        token=True,
    )
except Exception as exc:
    if modelo_base_id == modelo_fallback:
        raise RuntimeError(
            f"Não foi possível carregar o modelo {modelo_base_id}. Verifique o acesso ao Hugging Face e o modelo escolhido. Detalhes: {exc}"
        ) from exc

    print(f"\n[AVISO] O modelo principal '{modelo_base_id}' não está acessível ({exc})")
    print(f"Tentando usar o modelo público de fallback: {modelo_fallback}")

    tokenizer = AutoTokenizer.from_pretrained(modelo_fallback)
    model = AutoModelForCausalLM.from_pretrained(
        modelo_fallback,
        dtype=torch.bfloat16,
        device_map={"": 0},
    )
    modelo_base_id = modelo_fallback

print(f"\nInjetando os pesos treinados (LoRA) de: {caminho_modelo_final}...")
model = PeftModel.from_pretrained(model, caminho_modelo_final)
print("Modelo pronto para uso!\n")

perguntas = [
    "### Pergunta: O que é saúde pública?\n### Resposta:",
    "### Pergunta: Qual o principal objetivo da etapa de triagem em um hospital?\n### Resposta:",
   
]

for i, prompt in enumerate(perguntas, 1):
    print(f"--- TESTE {i} ---")
    print(f"Prompt: {prompt}")
    
    # Envia a pergunta explicitamente para a placa de vídeo
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda") 

    outputs = model.generate(
        **inputs,
        max_new_tokens=150, 
        do_sample=False,
        use_cache=True, 
        pad_token_id=tokenizer.eos_token_id
    )

    resposta_texto = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print(f"Gerado:\n{resposta_texto}\n")
    print("=" * 50 + "\n")