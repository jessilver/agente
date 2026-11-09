import os
import threading
import torch
import customtkinter as ctk
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configuração visual do aplicativo
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# =========================================================================================
# 🧠 PARTE 1: INTELIGÊNCIA ARTIFICIAL E RAG (Processamento, Leis e LLaMA)
# =========================================================================================
class MotorIA:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.encoder_rag = None
        self.collection = None
        self.pronto = False

    def carregar_motores(self):
        """Carrega o banco de leis e o modelo LLaMA na memória RAM."""
        diretorio_script = os.path.dirname(os.path.abspath(__file__))
        caminho_chroma = os.path.join(diretorio_script, "chroma_db_sus")
        modelo_base_id = "meta-llama/Llama-3.2-3B-Instruct"

        # 1. Conecta ao Banco Vetorial Local (ChromaDB)
        client_chroma = chromadb.PersistentClient(path=caminho_chroma)
        self.collection = client_chroma.get_or_create_collection(name="base_juridica_sus")
        
        # 2. Carrega o modelo de buscas semânticas (RAG)
        self.encoder_rag = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

        # 3. Carrega o modelo de linguagem (LLaMA)
        self.tokenizer = AutoTokenizer.from_pretrained(modelo_base_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            modelo_base_id,
            dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        self.pronto = True

    def analisar_texto(self, texto_usuario):
        """Analisa dinamicamente o que o usuário digitou e gera a resposta adequada."""
        if not self.pronto:
            return "Erro: A Inteligência Artificial ainda não foi carregada."

        # 1. Busca no RAG baseada no que o USUÁRIO digitou (Busca Dinâmica)
        query_embedding = self.encoder_rag.encode([texto_usuario]).tolist()
        resultados = self.collection.query(query_embeddings=query_embedding, n_results=3)

        contexto_legal_recuperado = ""
        if resultados and resultados.get("documents") and len(resultados["documents"][0]) > 0:
            for i, doc in enumerate(resultados["documents"][0]):
                fonte = resultados["metadatas"][0][i].get("fonte", "Base de Leis")
                contexto_legal_recuperado += f"\n--- Trecho Legal ({fonte}) ---\n{doc}\n"

        # 2. Prompt Inteligente: Sabe diferenciar um "Oi" de uma "Portaria"
        system_prompt = (
            "Você é o 'Agente Auditor do SUS', um especialista rigoroso na legislação de saúde pública do Brasil (Constituição, Lei 8.080, etc). "
            "Você deve seguir estas 3 regras INCONDICIONALMENTE:\n\n"
            "REGRA 1 - SAUDAÇÕES: Se o usuário enviar apenas 'oi', 'olá', 'quem é você', apresente-se educadamente e pare por aí.\n\n"
            "REGRA 2 - PERGUNTAS DIRETAS: Se o usuário fizer uma pergunta (ex: 'Existe lei para X?'), responda diretamente BASEANDO-SE EXCLUSIVAMENTE no [CONTEXTO JURÍDICO] fornecido. NUNCA invente nomes de leis, números ou artigos. Se o contexto estiver vazio ou não responder à pergunta, diga: 'Não encontrei essa informação na minha base de leis local.' Não use o formato de Veredito para perguntas.\n\n"
            "REGRA 3 - AUDITORIA DE DOCUMENTOS: Apenas se o usuário enviar uma Portaria, Decreto, ou ato administrativo para análise, emita um Parecer Técnico estruturado com: 1. Veredito (Conforme/Inconforme), 2. Fundamentação Legal e 3. Recomendação."
        )

        user_content = (
            f"[CONTEXTO JURÍDICO RECUPERADO DA BASE DE LEIS (SUS / TO)]\n"
            f"{contexto_legal_recuperado}\n\n"
            f"[MENSAGEM OU DOCUMENTO DO USUÁRIO]\n"
            f"{texto_usuario}"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        prompt_formatado = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # 3. Geração da Resposta pelo LLaMA
        inputs = self.tokenizer(prompt_formatado, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=400,
                do_sample=False,
                repetition_penalty=1.15,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        resposta_completa = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        resposta_final = resposta_completa.split("assistant")[-1].strip()
        
        return resposta_final


# =========================================================================================
# 🖥️ PARTE 2: INTERFACE GRÁFICA (Telas, Botões e Eventos)
# =========================================================================================
class InterfaceGrafica(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Agente de Auditoria de Conformidade do SUS")
        self.geometry("900x700")

        # Instancia a IA (mas ainda não a carrega na RAM)
        self.ia = MotorIA()

        self.construir_tela()

    def construir_tela(self):
        """Desenha os botões e caixas de texto na tela."""
        # Título
        self.title_label = ctk.CTkLabel(
            self, text="Agente Auditor de Conformidade do SUS", font=ctk.CTkFont(size=20, weight="bold")
        )
        self.title_label.pack(pady=10)

        # Botão Ligar Motores
        self.btn_carregar = ctk.CTkButton(
            self, text="1. Ligar Inteligência Artificial (Carregar RAM)", command=self.evento_carregar_ia
        )
        self.btn_carregar.pack(pady=5)

        # Label de Status
        self.status_label = ctk.CTkLabel(
            self, text="Status: Aguardando inicialização...", font=ctk.CTkFont(size=12, slant="italic")
        )
        self.status_label.pack(pady=5)

        # Caixa de Entrada do Usuário
        self.lbl_entrada = ctk.CTkLabel(
            self, text="Digite sua mensagem, portaria ou documento abaixo:", font=ctk.CTkFont(size=14, weight="bold")
        )
        self.lbl_entrada.pack(anchor="w", padx=20, pady=(10, 0))

        self.txt_entrada = ctk.CTkTextbox(self, height=150)
        self.txt_entrada.pack(fill="x", padx=20, pady=5)
        self.txt_entrada.insert("1.0", "Oi, quem é você?") # Exemplo de teste inicial

        # Botão Enviar
        self.btn_analisar = ctk.CTkButton(
            self, text="2. Enviar para Análise", command=self.evento_analisar_texto, state="disabled", fg_color="green"
        )
        self.btn_analisar.pack(pady=15)

        # Caixa de Resposta da IA
        self.lbl_saida = ctk.CTkLabel(
            self, text="Resposta da IA / Parecer Técnico:", font=ctk.CTkFont(size=14, weight="bold")
        )
        self.lbl_saida.pack(anchor="w", padx=20, pady=(5, 0))

        self.txt_saida = ctk.CTkTextbox(self, height=220)
        self.txt_saida.pack(fill="both", expand=True, padx=20, pady=10)

    # --- Eventos de Clique (Executados em Threads para não travar a tela) ---
    def evento_carregar_ia(self):
        self.btn_carregar.configure(state="disabled")
        self.status_label.configure(text="Status: Carregando modelo e banco de leis na RAM... (Aguarde)")
        threading.Thread(target=self._tarefa_carregar_ia, daemon=True).start()

    def _tarefa_carregar_ia(self):
        try:
            self.ia.carregar_motores()
            self.status_label.configure(text="Status: IA e Banco de Leis Prontos!")
            self.btn_analisar.configure(state="normal")
        except Exception as e:
            self.status_label.configure(text=f"Erro ao carregar IA: {e}")

    def evento_analisar_texto(self):
        self.btn_analisar.configure(state="disabled")
        self.status_label.configure(text="Status: Analisando e gerando resposta...")
        self.txt_saida.delete("1.0", "end")
        threading.Thread(target=self._tarefa_analisar_texto, daemon=True).start()

    def _tarefa_analisar_texto(self):
        try:
            texto_digitado = self.txt_entrada.get("1.0", "end").strip()
            if not texto_digitado:
                self.txt_saida.insert("1.0", "Por favor, digite alguma coisa na caixa acima.")
                return

            # Chama a IA enviando o texto real do usuário
            resposta = self.ia.analisar_texto(texto_digitado)
            
            # Mostra a resposta na tela
            self.txt_saida.insert("1.0", resposta)
            self.status_label.configure(text="Status: Resposta concluída!")
        except Exception as e:
            self.txt_saida.insert("1.0", f"Erro na análise: {e}")
            self.status_label.configure(text="Status: Falha na geração.")
        finally:
            self.btn_analisar.configure(state="normal")


# =========================================================================================
# 🚀 PONTO DE ENTRADA DO APLICATIVO
# =========================================================================================
if __name__ == "__main__":
    app = InterfaceGrafica()
    app.mainloop()