import argparse
from datetime import datetime
import glob
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer
import yaml

# ==========================================
# CONFIGURAÇÕES GERAIS E HIPERPARÂMETROS
# ==========================================
NOME_COLECAO = "base_juridica_sus"
MODELO_EMBEDDING = "all-MiniLM-L6-v2"
DISPOSITIVO = "cpu"

BATCH_EMBEDDING = 256
MAX_WORKERS = 6
BATCH_ARQUIVOS = 64  # Reduzido para dar feedback visual mais frequente no terminal
BATCH_CHROMA = 1000


def configurar_torch():
    """Limita as threads internas do PyTorch para otimizar o desempenho na CPU."""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    try:
        import torch

        torch.set_num_threads(6)
        try:
            torch.set_num_interop_threads(2)
        except RuntimeError:
            pass
    except Exception:
        pass


def sha256_texto(texto: str) -> str:
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


def id_estavel(
    caminho_relativo: str,
    dispositivo: str,
    indice_chunk: int,
    texto: str,
    ocorrencia: int = 0,
) -> str:
    """Gera um ID determinístico para cada trecho legal no ChromaDB."""
    hash_conteudo = sha256_texto(texto)
    chave = f"{caminho_relativo}|{dispositivo}|{indice_chunk}|{hash_conteudo}|{ocorrencia}"
    return hashlib.sha256(chave.encode("utf-8")).hexdigest()


class Manifesto:
    """Gerencia o cache SQLite local para impedir reindexação desnecessária de arquivos inalterados."""

    def __init__(self, caminho: Path):
        caminho.parent.mkdir(parents=True, exist_ok=True)
        self.conexao = sqlite3.connect(str(caminho))
        self.conexao.execute(
            """CREATE TABLE IF NOT EXISTS arquivos (
                caminho_relativo TEXT PRIMARY KEY,
                hash_arquivo TEXT NOT NULL,
                ids_json TEXT NOT NULL,
                atualizado_em TEXT NOT NULL
            )"""
        )
        self.conexao.commit()

    def obter(self, relativo: str):
        linha = self.conexao.execute(
            "SELECT hash_arquivo, ids_json FROM arquivos WHERE caminho_relativo = ?",
            (relativo,),
        ).fetchone()
        if not linha:
            return None
        return linha[0], json.loads(linha[1])

    def salvar(self, relativo: str, hash_arquivo: str, ids: list[str]):
        self.conexao.execute(
            """INSERT INTO arquivos(caminho_relativo, hash_arquivo, ids_json, atualizado_em)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(caminho_relativo) DO UPDATE SET
                 hash_arquivo=excluded.hash_arquivo,
                 ids_json=excluded.ids_json,
                 atualizado_em=excluded.atualizado_em""",
            (
                relativo,
                hash_arquivo,
                json.dumps(ids),
                datetime.now().isoformat(timespec="seconds"),
            ),
        )

    def commit(self):
        self.conexao.commit()

    def fechar(self):
        self.conexao.close()


# ==========================================
# LEITURA E PROCESSAMENTO DE ARQUIVOS
# ==========================================
def ler_arquivo(caminho: Path) -> str:
    try:
        return caminho.read_text(encoding="utf-8", errors="ignore").strip()
    except Exception:
        return ""


def separar_frontmatter(conteudo: str) -> tuple[dict[str, Any], str]:
    """Separa o cabeçalho YAML do texto e limpa caracteres de controle que causam erros."""
    if not conteudo.startswith("---"):
        return {}, conteudo

    partes = conteudo.split("---", 2)
    if len(partes) < 3:
        return {}, conteudo

    yaml_bruto = partes[1]
    corpo = partes[2].strip()

    yaml_limpo = re.sub(
        r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", yaml_bruto
    )

    try:
        dados = yaml.safe_load(yaml_limpo) or {}
    except Exception:
        dados = {}

    return dados, corpo


def criar_chunks(
    corpo: str, tamanho_chunk: int = 500
) -> list[tuple[str, str, int]]:
    """Divide o texto em blocos de até 500 caracteres."""
    segmentos = []
    if not corpo:
        return segmentos

    paragrafos = [p.strip() for p in corpo.split("\n\n") if p.strip()]
    idx = 0

    for i, p in enumerate(paragrafos):
        dispositivo_rotulo = f"Trecho {i + 1}"
        if len(p) <= tamanho_chunk:
            segmentos.append((dispositivo_rotulo, p, idx))
            idx += 1
        else:
            for j in range(0, len(p), tamanho_chunk):
                fatia = p[j : j + tamanho_chunk].strip()
                if len(fatia) > 30:
                    segmentos.append((f"{dispositivo_rotulo} (parte)", fatia, idx))
                    idx += 1
    return segmentos


def classificar_documento(
    caminho: Path, pasta_leis: Path
) -> dict[str, str]:
    caminho_rel = caminho.relative_to(pasta_leis).as_posix().lower()
    esfera = "federal"
    uf = "BR"

    if "estaduais" in caminho_rel or "/to/" in caminho_rel or "tocantins" in caminho_rel:
        esfera = "estadual"
        uf = "TO"
    elif "constituicao" in caminho_rel:
        esfera = "constitucional"

    return {"esfera": esfera, "uf": uf}


def montar_metadados(
    caminho: Path,
    pasta_leis: Path,
    yaml_dados: dict[str, Any],
    classificacao: dict[str, str],
    dispositivo: str,
    texto_chunk: str,
) -> dict[str, Any]:
    relativo = caminho.relative_to(pasta_leis).as_posix()
    return {
        "fonte": caminho.name,
        "caminho_relativo": relativo,
        "esfera": classificacao["esfera"],
        "uf": classificacao["uf"],
        "tipo": str(yaml_dados.get("tipo", "Norma")),
        "numero": str(yaml_dados.get("numero", "")),
        "dispositivo": dispositivo,
    }


def preparar_arquivo(caminho: Path, pasta_leis: Path) -> dict[str, Any]:
    """Leitura de arquivo e chunking executados pelas threads em paralelo."""
    conteudo = ler_arquivo(caminho)
    if not conteudo:
        return {"caminho": caminho, "vazio": True}

    yaml_dados, corpo = separar_frontmatter(conteudo)
    if not corpo:
        return {"caminho": caminho, "vazio": True}

    relativo = caminho.relative_to(pasta_leis).as_posix()
    segmentos = criar_chunks(corpo)
    classificacao = classificar_documento(caminho, pasta_leis)

    return {
        "caminho": caminho,
        "relativo": relativo,
        "hash_arquivo": sha256_texto(conteudo),
        "yaml": yaml_dados,
        "classificacao": classificacao,
        "segmentos": segmentos,
    }


# ==========================================
# EXECUÇÃO PRINCIPAL E INDEXAÇÃO
# ==========================================
def coletar_arquivos(pasta_leis: Path) -> list[Path]:
    arquivos = []
    for ext in ("*.md", "*.txt"):
        arquivos.extend(pasta_leis.rglob(ext))
    return sorted(arquivos)


def abrir_colecao(caminho_banco: Path, reconstruir: bool = False):
    client = chromadb.PersistentClient(path=str(caminho_banco))
    if reconstruir:
        try:
            client.delete_collection(NOME_COLECAO)
        except Exception:
            pass
    colecao = client.get_or_create_collection(name=NOME_COLECAO)
    return client, colecao


def indexar(
    pasta_leis: Path,
    caminho_banco: Path,
    reconstruir: bool = False,
    workers: int = MAX_WORKERS,
    batch_arquivos: int = BATCH_ARQUIVOS,
    batch_chroma: int = BATCH_CHROMA,
    batch_embedding: int = BATCH_EMBEDDING,
    forcar: bool = False,
    max_arquivos: int | None = None,
) -> None:
    arquivos = coletar_arquivos(pasta_leis)
    if max_arquivos:
        arquivos = arquivos[:max_arquivos]

    print(f"\nPasta de leis: {pasta_leis}", flush=True)
    print(f"Banco ChromaDB: {caminho_banco}", flush=True)
    print(f"Coleção: {NOME_COLECAO}", flush=True)
    print(f"Arquivos localizados: {len(arquivos)}", flush=True)
    print(
        f"Paralelismo: {workers} threads | Lote de arquivos: {batch_arquivos} | "
        f"Embeddings: {batch_embedding} | ChromaDB: {batch_chroma}\n",
        flush=True,
    )

    if not arquivos:
        print("[AVISO] Nenhum arquivo .md ou .txt encontrado.", flush=True)
        return

    configurar_torch()
    print(f"Carregando modelo de embeddings: {MODELO_EMBEDDING} (CPU)...", flush=True)
    encoder = SentenceTransformer(MODELO_EMBEDDING, device=DISPOSITIVO)
    _, colecao = abrir_colecao(caminho_banco, reconstruir)

    manifesto = Manifesto(caminho_banco / "indexador_manifesto.sqlite3")
    processados = 0
    ignorados = 0
    erros = 0
    total_chunks = 0
    total_lotes = (len(arquivos) + batch_arquivos - 1) // batch_arquivos

    def iterar_lotes(lista, tamanho):
        for inicio in range(0, len(lista), tamanho):
            yield lista[inicio : inicio + tamanho]

    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for numero_lote, grupo in enumerate(
                iterar_lotes(arquivos, batch_arquivos), start=1
            ):
                print(
                    f"-> [Lote {numero_lote}/{total_lotes}] Processando {len(grupo)} arquivos...",
                    flush=True,
                )

                futuros = list(
                    executor.map(lambda p: preparar_arquivo(p, pasta_leis), grupo)
                )

                arquivos_lote = []
                registros = []
                ids_por_arquivo: dict[str, list[str]] = {}
                antigos_por_arquivo: dict[str, list[str]] = {}
                hashes_por_arquivo: dict[str, str] = {}

                for item in futuros:
                    caminho = item["caminho"]
                    if item.get("vazio"):
                        ignorados += 1
                        continue

                    relativo = item["relativo"]
                    hash_arquivo = item["hash_arquivo"]
                    estado = manifesto.obter(relativo)

                    if estado and estado[0] == hash_arquivo and not forcar:
                        ignorados += 1
                        continue

                    yaml_dados = item["yaml"]
                    classificacao = item["classificacao"]
                    ids_novos = []
                    ocorrencias_ids: dict[tuple[str, int, str], int] = {}

                    for dispositivo, texto_chunk, indice_chunk in item["segmentos"]:
                        hash_chunk = sha256_texto(texto_chunk)
                        chave_base = (dispositivo, indice_chunk, hash_chunk)
                        ocorrencia = ocorrencias_ids.get(chave_base, 0)
                        ocorrencias_ids[chave_base] = ocorrencia + 1

                        identificador = id_estavel(
                            relativo,
                            dispositivo,
                            indice_chunk,
                            texto_chunk,
                            ocorrencia,
                        )
                        titulo = yaml_dados.get("titulo", caminho.stem)
                        ementa = yaml_dados.get("ementa", "")

                        documento = (
                            f"Norma: {titulo}\n"
                            f"Tipo: {yaml_dados.get('tipo', 'Não informado')}\n"
                            f"Número: {yaml_dados.get('numero', 'Não informado')}\n"
                            f"Data: {yaml_dados.get('data', 'Não informada')}\n"
                            f"Esfera: {classificacao['esfera']}\n"
                            f"Dispositivo: {dispositivo}\n"
                            f"Ementa: {ementa}\n\n"
                            f"Texto do dispositivo:\n{texto_chunk}"
                        )

                        metadados = montar_metadados(
                            caminho,
                            pasta_leis,
                            yaml_dados,
                            classificacao,
                            dispositivo,
                            texto_chunk,
                        )
                        metadados["indice_chunk"] = indice_chunk

                        registros.append(
                            {
                                "id": identificador,
                                "documento": documento,
                                "metadados": metadados,
                            }
                        )
                        ids_novos.append(identificador)

                    if estado:
                        ids_antigos = estado[1]
                    else:
                        antigos = colecao.get(
                            where={"caminho_relativo": relativo},
                            include=["metadatas"],
                        )
                        ids_antigos = antigos.get("ids", [])

                    ids_por_arquivo[relativo] = ids_novos
                    antigos_por_arquivo[relativo] = ids_antigos
                    hashes_por_arquivo[relativo] = hash_arquivo
                    arquivos_lote.append(relativo)

                lote_ok = True
                if registros:
                    for inicio in range(0, len(registros), batch_embedding):
                        sub = registros[inicio : inicio + batch_embedding]
                        try:
                            embeddings = encoder.encode(
                                [r["documento"] for r in sub],
                                batch_size=batch_embedding,
                                normalize_embeddings=True,
                                show_progress_bar=False,
                                convert_to_numpy=True,
                            ).tolist()

                            for pos in range(0, len(sub), batch_chroma):
                                sub_registros = sub[pos : pos + batch_chroma]
                                sub_embeddings = embeddings[pos : pos + batch_chroma]
                                colecao.upsert(
                                    ids=[r["id"] for r in sub_registros],
                                    documents=[r["documento"] for r in sub_registros],
                                    embeddings=sub_embeddings,
                                    metadatas=[r["metadados"] for r in sub_registros],
                                )
                        except Exception as erro:
                            lote_ok = False
                            erros += 1
                            print(
                                f"[ERRO] Falha ao gerar embeddings no lote {numero_lote}: {erro}",
                                flush=True,
                            )
                            break

                if lote_ok:
                    for relativo in arquivos_lote:
                        ids_novos = ids_por_arquivo[relativo]
                        ids_antigos = set(antigos_por_arquivo[relativo])
                        ids_remover = ids_antigos - set(ids_novos)
                        if ids_remover:
                            colecao.delete(ids=list(ids_remover))
                        manifesto.salvar(
                            relativo, hashes_por_arquivo[relativo], ids_novos
                        )
                        processados += 1
                        total_chunks += len(ids_novos)
                    manifesto.commit()
                else:
                    erros += max(0, len(arquivos_lote) - 1)

                percorridos = min(numero_lote * batch_arquivos, len(arquivos))
                print(
                    f"   V [Concluído {percorridos}/{len(arquivos)} arquivos] "
                    f"Novos/Atualizados: {processados} | Inalterados: {ignorados} | "
                    f"Chunks salvos: {total_chunks} | Erros: {erros}\n",
                    flush=True,
                )

    finally:
        manifesto.fechar()

    print("========== INDEXAÇÃO FINALIZADA ==========", flush=True)
    print(f"Arquivos processados/atualizados: {processados}", flush=True)
    print(f"Arquivos ignorados/inalterados: {ignorados}", flush=True)
    print(f"Erros: {erros}", flush=True)
    print(f"Total de registros na coleção: {colecao.count()}", flush=True)


if __name__ == "__main__":
    diretorio_script = Path(__file__).resolve().parent
    pasta_leis_padrao = diretorio_script / "data" / "leis"
    banco_padrao = diretorio_script / "chroma_db_sus"

    parser = argparse.ArgumentParser(
        description="Indexador RAG de Leis do SUS com suporte a cache SQLite e multithreading."
    )
    parser.add_argument("--base", type=Path, default=pasta_leis_padrao)
    parser.add_argument("--banco", type=Path, default=banco_padrao)
    parser.add_argument(
        "--reconstruir",
        action="store_true",
        help="Recria a coleção do banco vetorial do zero.",
    )
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--batch-arquivos", type=int, default=BATCH_ARQUIVOS)
    parser.add_argument("--batch-chroma", type=int, default=BATCH_CHROMA)
    parser.add_argument("--batch-embedding", type=int, default=BATCH_EMBEDDING)
    parser.add_argument(
        "--forcar",
        action="store_true",
        help="Reprocessa todos os arquivos ignorando o cache.",
    )
    parser.add_argument(
        "--max-arquivos",
        type=int,
        help="Limita o número de arquivos para testes rápidos.",
    )

    args = parser.parse_args()

    if min(args.workers, args.batch_arquivos, args.batch_chroma, args.batch_embedding) < 1:
        parser.error("Os valores dos lotes e workers devem ser maiores que zero.")

    indexar(
        args.base,
        args.banco,
        args.reconstruir,
        workers=args.workers,
        batch_arquivos=args.batch_arquivos,
        batch_chroma=args.batch_chroma,
        batch_embedding=args.batch_embedding,
        forcar=args.forcar,
        max_arquivos=args.max_arquivos,
    )