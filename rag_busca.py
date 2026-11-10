"""Busca de trechos legais no ChromaDB (somente leitura).

Etapas: busca vetorial + busca por palavra-chave -> descarte de arquivos de trabalho (backups, auditorias,
índices) -> limpeza de HTML -> remoção de duplicados -> metadados corrigidos -> reranker -> nota mínima.
Inclui verificação das citações da resposta contra os trechos recuperados.
"""

import hashlib
import html
import re
import unicodedata

ESFERAS_PRIORITARIAS = ["constitucional", "federal"]
MODELO_RERANKER = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
N_CANDIDATOS_RERANK = 50  # por busca (esferas prioritárias, base inteira e cada palavra-chave)
N_CANDIDATOS_PALAVRA = 15  # por palavra-chave
# Nota do cross-encoder abaixo da qual o trecho é considerado irrelevante e não vai para o prompt.
# Calibrada nos testes de 10/11/2026: Art. 196 (relevante) ≈ -3,7; trechos sem relação ≈ -5,3 ou menos.
NOTA_MINIMA = -5.0

# Pastas de trabalho do acervo que foram indexadas junto com as leis (cópias, auditorias, índices).
_SEGMENTOS_DESCARTADOS = ("_fonte", "_indice", "auditorias", "backup", "candidatos", "_recuperacao")

_RE_TAG = re.compile(r"<[^>]*>")
_RE_TAG_CORTADA = re.compile(r"^[^<>]*>|<[^<>]*$")
_RE_ESPACOS = re.compile(r"[ \t ]+")
_RE_LINHAS = re.compile(r"\n\s*\n+")
_RE_CITACAO_LEI = re.compile(
    r"\blei(?:\s+(?:complementar|ordin[áa]ria|org[âa]nica))?\s*(?:n[º°o]\.?\s*)?(\d{1,2})\.?(\d{3})\b", re.I
)
_RE_NORMA = re.compile(r"^Norma: (.*)$", re.M)
_RE_NUMERO_NORMA = re.compile(r"n[º°o]\.?\s*([\d.]+)", re.I)
_RE_UF = re.compile(r"^estaduais/([A-Z]{2})/")

_cache_consultas = {}


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


def _descartar(metadados) -> bool:
    caminho = metadados.get("caminho_relativo", "")
    return any(segmento in caminho for segmento in _SEGMENTOS_DESCARTADOS)


def corrigir_metadados(metadados, documento):
    """Corrige na leitura a UF (pelo caminho) e o número da norma (pelo cabeçalho 'Norma:')."""
    caminho = metadados.get("caminho_relativo", "")
    uf = metadados.get("uf", "")
    m_uf = _RE_UF.match(caminho)
    if m_uf:
        uf = m_uf.group(1)
    norma = ""
    numero = metadados.get("numero", "")
    m_norma = _RE_NORMA.search(documento)
    if m_norma:
        norma = m_norma.group(1).strip()
        m_num = _RE_NUMERO_NORMA.search(norma)
        if m_num:
            numero = m_num.group(1).replace(".", "").strip()
    return {"uf": uf, "numero": numero, "norma": norma}


def extrair_numeros_lei(texto):
    """'Lei 8080' / 'Lei 8.080' -> ['8.080'] (números de 4-5 dígitos, formato com ponto usado nas leis)."""
    vistos = []
    for a, b in _RE_CITACAO_LEI.findall(texto):
        numero = f"{a}.{b}"
        if numero not in vistos:
            vistos.append(numero)
    return vistos


def _consultar(collection, embedding, n_resultados, filtro=None, filtro_texto=None):
    try:
        r = collection.query(
            query_embeddings=embedding, n_results=n_resultados, where=filtro, where_document=filtro_texto
        )
    except Exception as erro:
        print(f"Erro na consulta RAG (filtro={filtro}, texto={filtro_texto}): {erro}")
        return []
    if not r or not r.get("documents") or not r["documents"][0]:
        return []
    return list(zip(r["documents"][0], r["metadatas"][0], r["distances"][0]))


def carregar_reranker(dispositivo="cpu"):
    """Cross-encoder multilíngue que reordena os candidatos pela relevância real para a consulta."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(MODELO_RERANKER, device=dispositivo, max_length=512)


def aquecer(collection, encoder):
    """Faz uma consulta descartável para carregar o índice da base em memória antes do primeiro uso."""
    _consultar(collection, encoder.encode(["saúde"]).tolist(), 1)


def buscar_trechos(
    collection,
    encoder,
    consulta,
    n_final=3,
    n_candidatos=20,
    esferas=ESFERAS_PRIORITARIAS,
    reranker=None,
    consulta_rerank=None,
    palavras_chave=None,
    nota_minima=NOTA_MINIMA,
    usar_cache=True,
):
    """Retorna até n_final trechos distintos, limpos e (com reranker) acima da nota mínima.

    - Busca vetorial nas esferas prioritárias e na base inteira.
    - Busca por palavra-chave: os números de lei citados na consulta ('8.080') e as palavras_chave
      informadas, ordenadas pela similaridade vetorial entre os trechos que contêm o termo.
    - Descarta arquivos de trabalho do acervo (backups, auditorias, índices), limpa HTML e remove duplicados.
    - Com reranker: ordena pela nota do cross-encoder (usando consulta_rerank, se informada) e descarta
      os trechos abaixo de nota_minima; pode devolver lista vazia quando nada é relevante.
    Cada item: 'texto', 'fonte', 'esfera', 'uf', 'numero', 'norma', 'distancia', 'nota', 'origem'.
    A base é só lida: nenhum dado do ChromaDB é alterado.
    """
    termos = list(dict.fromkeys(extrair_numeros_lei(consulta) + list(palavras_chave or [])))
    chave_cache = (consulta, consulta_rerank, n_final, n_candidatos, tuple(esferas or ()), tuple(termos),
                   reranker is not None, nota_minima)
    if usar_cache and chave_cache in _cache_consultas:
        return [dict(t) for t in _cache_consultas[chave_cache]]

    embedding = encoder.encode([consulta]).tolist()

    candidatos = []
    if esferas:
        candidatos += [(*c, "vetorial-prioritaria") for c in
                       _consultar(collection, embedding, n_candidatos, {"esfera": {"$in": list(esferas)}})]
    candidatos += [(*c, "vetorial") for c in _consultar(collection, embedding, n_candidatos)]
    for termo in termos:
        candidatos += [(*c, f"palavra:{termo}") for c in
                       _consultar(collection, embedding, N_CANDIDATOS_PALAVRA, filtro_texto={"$contains": termo})]

    vistos = set()
    trechos = []
    for documento, metadados, distancia, origem in candidatos:
        if _descartar(metadados):
            continue
        texto = limpar_html(documento)
        chave = _chave(texto)
        if not texto or chave in vistos:
            continue
        vistos.add(chave)
        trechos.append({
            "texto": texto,
            "fonte": metadados.get("fonte", "Base de Leis"),
            "esfera": metadados.get("esfera", ""),
            **corrigir_metadados(metadados, documento),
            "distancia": distancia,
            "nota": None,
            "origem": origem,
        })

    if reranker is not None and trechos:
        pergunta = consulta_rerank or consulta
        notas = reranker.predict([(pergunta, t["texto"]) for t in trechos], show_progress_bar=False)
        for t, nota in zip(trechos, notas):
            t["nota"] = float(nota)
        trechos.sort(key=lambda t: t["nota"], reverse=True)
        if nota_minima is not None:
            trechos = [t for t in trechos if t["nota"] >= nota_minima]

    resultado = trechos[:n_final]
    if usar_cache:
        _cache_consultas[chave_cache] = [dict(t) for t in resultado]
    return resultado


def formatar_contexto(trechos, rotulo="Artigo/Trecho Legal"):
    if not trechos:
        return "Nenhum artigo relevante foi encontrado na base local."
    partes = []
    for t in trechos:
        origem = t["fonte"] + (f", {t['uf']}" if t.get("esfera") == "estadual" and t.get("uf") else "")
        partes.append(f"\n--- {rotulo} ({origem}) ---\n{t['texto']}\n")
    return "".join(partes)


# Regra comum aos prompts: só citar o que está no contexto.
REGRA_CITACOES = (
    "Cite SOMENTE artigos, leis e trechos que aparecem literalmente no contexto jurídico fornecido, "
    "indicando a fonte entre parênteses. Nunca cite de memória artigos, números de lei ou alíneas que não "
    "estejam no contexto. Se o contexto não trouxer fundamento suficiente, diga explicitamente: "
    "'A base de leis local não trouxe fundamento suficiente para este ponto.'"
)

_RE_CITACAO_ARTIGO = re.compile(r"\bart(?:igo)?s?\.?\s*(\d+)\s*[º°o]?", re.I)


def _normalizar(texto):
    texto = unicodedata.normalize("NFKD", texto.lower())
    return "".join(c for c in texto if not unicodedata.combining(c))


def verificar_citacoes(resposta, trechos):
    """Lista as citações (artigos e leis) da resposta e diz se cada uma aparece nos trechos recuperados."""
    contexto = _normalizar(" ".join(t["texto"] for t in trechos))
    artigos_contexto = {int(n) for n in _RE_CITACAO_ARTIGO.findall(contexto)}
    leis_contexto = {a + b for a, b in _RE_CITACAO_LEI.findall(contexto)}

    citacoes = []
    for n in dict.fromkeys(_RE_CITACAO_ARTIGO.findall(resposta)):
        citacoes.append({"citacao": f"Art. {n}", "no_contexto": int(n) in artigos_contexto})
    for a, b in dict.fromkeys(_RE_CITACAO_LEI.findall(resposta)):
        citacoes.append({"citacao": f"Lei {a}.{b}", "no_contexto": a + b in leis_contexto})
    return citacoes


def anotar_citacoes(resposta, citacoes):
    """Acrescenta à resposta um aviso com as citações que não vieram dos trechos recuperados."""
    fora = [c["citacao"] for c in citacoes if not c["no_contexto"]]
    if not fora:
        return resposta
    return (
        resposta.rstrip()
        + "\n\n---\n⚠ Verificação automática: estas citações NÃO aparecem nos trechos recuperados da base "
        + "e podem ter sido geradas de memória pelo modelo: "
        + ", ".join(fora)
        + "."
    )
