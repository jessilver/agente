"""Busca de trechos legais no ChromaDB com limpeza de HTML, remoção de duplicados e prioridade por esfera."""

import hashlib
import html
import re

ESFERAS_PRIORITARIAS = ["constitucional", "federal"]
MODELO_RERANKER = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
N_CANDIDATOS_RERANK = 25  # por busca (esferas prioritárias + base inteira) => até 50 candidatos

_RE_TAG = re.compile(r"<[^>]*>")
_RE_TAG_CORTADA = re.compile(r"^[^<>]*>|<[^<>]*$")
_RE_ESPACOS = re.compile(r"[ \t ]+")
_RE_LINHAS = re.compile(r"\n\s*\n+")


def limpar_html(texto: str) -> str:
    """Remove tags (inclusive as cortadas no início/fim do chunk), entidades e espaços repetidos."""
    texto = _RE_TAG.sub(" ", texto)
    texto = "\n".join(_RE_TAG_CORTADA.sub(" ", linha) for linha in texto.split("\n"))
    texto = html.unescape(texto)
    texto = _RE_ESPACOS.sub(" ", texto)
    texto = _RE_LINHAS.sub("\n\n", texto)
    return "\n".join(linha.strip() for linha in texto.split("\n")).strip()


def _chave(texto: str) -> str:
    return hashlib.sha256(" ".join(texto.lower().split()).encode("utf-8")).hexdigest()


def _consultar(collection, embedding, n_resultados, filtro):
    try:
        r = collection.query(query_embeddings=embedding, n_results=n_resultados, where=filtro)
    except Exception as erro:
        print(f"Erro na consulta RAG (filtro={filtro}): {erro}")
        return []
    if not r or not r.get("documents") or not r["documents"][0]:
        return []
    return list(zip(r["documents"][0], r["metadatas"][0], r["distances"][0]))


def carregar_reranker(dispositivo="cpu"):
    """Cross-encoder multilíngue que reordena os candidatos pela relevância real para a consulta."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(MODELO_RERANKER, device=dispositivo, max_length=512)


def buscar_trechos(
    collection,
    encoder,
    consulta,
    n_final=3,
    n_candidatos=20,
    esferas=ESFERAS_PRIORITARIAS,
    reranker=None,
    consulta_rerank=None,
):
    """Retorna até n_final trechos distintos e limpos.

    Sem reranker: primeiro as esferas prioritárias, depois o restante da base, na ordem do ChromaDB.
    Com reranker: junta os candidatos das duas buscas e ordena todos pela nota do cross-encoder,
    usando consulta_rerank (frase em linguagem natural) quando informada, senão a própria consulta.
    Cada item é um dict com 'texto', 'fonte', 'esfera', 'distancia' e 'nota' (None sem reranker).
    A base é só lida: nenhum dado do ChromaDB é alterado.
    """
    embedding = encoder.encode([consulta]).tolist()

    candidatos = []
    if esferas:
        candidatos += _consultar(collection, embedding, n_candidatos, {"esfera": {"$in": list(esferas)}})
    candidatos += _consultar(collection, embedding, n_candidatos, None)

    vistos = set()
    trechos = []
    for documento, metadados, distancia in candidatos:
        texto = limpar_html(documento)
        chave = _chave(texto)
        if not texto or chave in vistos:
            continue
        vistos.add(chave)
        trechos.append({
            "texto": texto,
            "fonte": metadados.get("fonte", "Base de Leis"),
            "esfera": metadados.get("esfera", ""),
            "distancia": distancia,
            "nota": None,
        })

    if reranker is not None and trechos:
        pergunta = consulta_rerank or consulta
        notas = reranker.predict([(pergunta, t["texto"]) for t in trechos], show_progress_bar=False)
        for t, nota in zip(trechos, notas):
            t["nota"] = float(nota)
        trechos.sort(key=lambda t: t["nota"], reverse=True)

    return trechos[:n_final]


def formatar_contexto(trechos, rotulo="Artigo/Trecho Legal"):
    if not trechos:
        return "Nenhum artigo específico foi encontrado na base local."
    return "".join(f"\n--- {rotulo} ({t['fonte']}) ---\n{t['texto']}\n" for t in trechos)
