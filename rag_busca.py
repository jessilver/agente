"""Busca de trechos legais no ChromaDB com limpeza de HTML, remoção de duplicados e prioridade por esfera."""

import hashlib
import html
import re

ESFERAS_PRIORITARIAS = ["constitucional", "federal"]

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


def buscar_trechos(collection, encoder, consulta, n_final=3, n_candidatos=20, esferas=ESFERAS_PRIORITARIAS):
    """Retorna até n_final trechos distintos e limpos.

    Primeiro busca nas esferas prioritárias; se faltar trecho, completa com o restante da base.
    Cada item é um dict com 'texto', 'fonte', 'esfera' e 'distancia'.
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
        })
        if len(trechos) >= n_final:
            break
    return trechos


def formatar_contexto(trechos, rotulo="Artigo/Trecho Legal"):
    if not trechos:
        return "Nenhum artigo específico foi encontrado na base local."
    return "".join(f"\n--- {rotulo} ({t['fonte']}) ---\n{t['texto']}\n" for t in trechos)
