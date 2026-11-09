import os
import chromadb

diretorio_script = os.path.dirname(os.path.abspath(__file__))
caminho_chroma = os.path.join(diretorio_script, "chroma_db_sus")

print("=== INSPEÇÃO DA BASE VETORIAL CHROMADB ===")
print(f"Diretoria do banco: {caminho_chroma}")

client = chromadb.PersistentClient(path=caminho_chroma)

try:
    collection = client.get_collection(name="base_juridica_sus")
    total_registos = collection.count()
    
    print(f"\nStatus: Coleção 'base_juridica_sus' encontrada.")
    print(f"Total de trechos/artigos indexados: {total_registos}")

    if total_registos > 0:
        print("\n--- TESTE DE CONSULTA DE AMOSTRA (Procurando: 'gratuidade taxa') ---")
        resultados = collection.query(
            query_texts=["gratuidade taxa cobranca exames"],
            n_results=2
        )
        
        for i, doc in enumerate(resultados['documents'][0]):
            fonte = resultados['metadatas'][0][i].get('fonte', 'Desconhecida')
            print(f"\n[Resultado {i+1} | Fonte: {fonte}]")
            print(doc[:300] + "...\n" + "-"*40)
    else:
        print("\n[ATENÇÃO] O banco existe, mas o total de registos é 0.")
        print("Reexecuta o script 'python indexar_base.py' para carregar os ficheiros de lei.")

except Exception as e:
    print(f"\n[ERRO] Falha ao aceder à coleção: {e}")