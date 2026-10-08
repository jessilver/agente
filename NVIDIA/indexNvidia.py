# ==========================================
# Treinamento de LLaMA-3.2-3B com LoRA e otimizações para placas NVIDIA
# ==========================================

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

from dataset_utils import load_vigilancia_sanitaria_dataset

# ==========================================
# 0. CONFIGURAÇÃO DE CAMINHOS
# ==========================================
diretorio_script = os.path.dirname(os.path.abspath(__file__))

caminho_dataset = os.path.join(
    diretorio_script,
    "..",
    "data",
    "raw",
    "03_vigilancia_sanitaria.jsonl",
)
caminho_checkpoints = os.path.join(diretorio_script, "..", "training", "checkpoints_salvos")
caminho_modelo_final = os.path.join(diretorio_script, "..", "models", "lora_model_saude")

print("=== INICIANDO AMBIENTE OTIMIZADO PARA GPU NVIDIA ===")
print(f"Buscando dataset em: {caminho_dataset}")

if not torch.cuda.is_available():
    raise RuntimeError("GPU NVIDIA não detectada pelo PyTorch. Verifique drivers e instalação do CUDA.")

gpu_nome = torch.cuda.get_device_name(0)
print(f"GPU detectada: {gpu_nome}")
print(f"Versão CUDA do PyTorch: {torch.version.cuda}")

# ==========================================
# 1. CONFIGURAÇÃO DE MEMÓRIA (4-BITS NF4)
# ==========================================
usa_bf16 = torch.cuda.is_bf16_supported()
tipo_dado = torch.bfloat16 if usa_bf16 else torch.float16

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=tipo_dado,
    bnb_4bit_use_double_quant=True,
)

# ==========================================
# 2. CARREGAMENTO DO MODELO
# ==========================================
modelo_id = os.environ.get("MODEL_ID", "meta-llama/Llama-3.2-3B")
modelo_fallback = os.environ.get("FALLBACK_MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

try:
    print(f"\nCarregando tokenizer do {modelo_id}...")
    tokenizer = AutoTokenizer.from_pretrained(modelo_id, token=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\nCarregando modelo base na VRAM da GPU NVIDIA...")
    model = AutoModelForCausalLM.from_pretrained(
        modelo_id,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="sdpa",
        token=True,
    )
except Exception as exc:
    if modelo_id == modelo_fallback:
        raise RuntimeError(
            f"Não foi possível carregar o modelo {modelo_id}. Verifique o acesso ao Hugging Face e o modelo escolhido. Detalhes: {exc}"
        ) from exc

    print(f"\n[AVISO] O modelo principal '{modelo_id}' não está acessível ({exc})")
    print(f"Tentando usar o modelo público de fallback: {modelo_fallback}")

    tokenizer = AutoTokenizer.from_pretrained(modelo_fallback)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        modelo_fallback,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="sdpa",
    )
    modelo_id = modelo_fallback

# ==========================================
# 3. CONFIGURAÇÃO DO LORA
# ==========================================
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# ==========================================
# 4. DATASET E TREINAMENTO
# ==========================================
dataset = load_vigilancia_sanitaria_dataset(caminho_dataset)
print(f"Registros válidos carregados: {len(dataset)}")

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=SFTConfig(
        dataset_text_field="text",
        max_length=1024,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_steps=60,
        learning_rate=2e-4,
        logging_steps=1,
        output_dir=caminho_checkpoints,
        optim="adamw_torch",
        bf16=usa_bf16,
        fp16=not usa_bf16,
        report_to="none",
    ),
)

print("\nIniciando o treinamento na GPU NVIDIA...")
trainer.train()

# ==========================================
# 5. SALVANDO OS PESOS FINAIS
# ==========================================
trainer.model.save_pretrained(caminho_modelo_final)
print(f"\nTreinamento concluído e salvo com sucesso em: {caminho_modelo_final}")
