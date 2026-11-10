"""Testes do rag_busca.py sem a base real: usam uma coleção e um encoder falsos."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import rag_busca as rb  # noqa: E402


class EncoderFalso:
    def encode(self, textos):
        return np.zeros((len(textos), 3))


class ColecaoFalsa:
    """Devolve os documentos configurados e registra cada chamada a query()."""

    def __init__(self, documentos):
        self.documentos = documentos  # lista de (documento, metadados, distancia)
        self.chamadas = []

    def query(self, query_embeddings, n_results, where=None, where_document=None):
        self.chamadas.append({"where": where, "where_document": where_document})
        docs = self.documentos[:n_results]
        return {
            "documents": [[d for d, _, _ in docs]],
            "metadatas": [[m for _, m, _ in docs]],
            "distances": [[x for _, _, x in docs]],
        }


class RerankerFalso:
    """Dá a nota pela posição de uma palavra no texto, para controlar a ordem nos testes."""

    def __init__(self, notas_por_palavra):
        self.notas = notas_por_palavra

    def predict(self, pares, show_progress_bar=False):
        return [next((n for p, n in self.notas.items() if p in texto), -10.0) for _, texto in pares]


def doc(texto, caminho="federais/lei.md", esfera="federal", distancia=0.1):
    return (texto, {"fonte": os.path.basename(caminho), "caminho_relativo": caminho, "esfera": esfera, "uf": "TO"},
            distancia)


@pytest.fixture(autouse=True)
def limpar_cache():
    rb._cache_consultas.clear()
    yield
    rb._cache_consultas.clear()


# ---------- limpar_html ----------

def test_limpar_html_remove_tags_entidades_e_tags_cortadas():
    sujo = ': Arial,sans-serif;"><strong>Art. 4°&nbsp;</strong>A política   de saúde</span></p><p style="mar'
    limpo = rb.limpar_html(sujo)
    assert "<" not in limpo and ">" not in limpo and "&nbsp;" not in limpo
    assert "Art. 4° A política de saúde" in limpo


# ---------- extrair_numeros_lei ----------

@pytest.mark.parametrize(
    "texto, esperado",
    [
        ("Lei 8080 artigo 7 196", ["8.080"]),
        ("Lei nº 8.080/1990 e Lei 8.142", ["8.080", "8.142"]),
        ("Lei Complementar nº 141, de 2012", []),  # 3 dígitos não é número de lei federal desse formato
        ("artigo 196 de 1990", []),  # anos e artigos não são números de lei
    ],
)
def test_extrair_numeros_lei(texto, esperado):
    assert rb.extrair_numeros_lei(texto) == esperado


# ---------- corrigir_metadados ----------

def test_corrigir_metadados_tira_uf_do_caminho_e_numero_do_cabecalho():
    meta = {"caminho_relativo": "estaduais/AC/leis/l-2680-2013.md", "uf": "TO", "numero": ""}
    corrigido = rb.corrigir_metadados(meta, "Norma: Lei nº 2.680, de 02/01/2013\nTipo: Lei")
    assert corrigido == {"uf": "AC", "numero": "2680", "norma": "Lei nº 2.680, de 02/01/2013"}


def test_corrigir_metadados_mantem_valores_sem_padrao():
    meta = {"caminho_relativo": "constituicao.md", "uf": "BR", "numero": ""}
    assert rb.corrigir_metadados(meta, "Texto sem cabeçalho") == {"uf": "BR", "numero": "", "norma": ""}


# ---------- buscar_trechos ----------

def test_buscar_trechos_descarta_backups_e_duplicados():
    col = ColecaoFalsa([
        doc("Art. 196. A saúde é direito de todos."),
        doc("Art. 196.  A saúde é   direito de todos."),  # mesmo texto com espaços diferentes
        doc("Art. 196. A saúde é direito de todos (cópia)", caminho="estaduais/AC/_fonte/auditorias/x/backup/a.md"),
        doc("Art. 5. Todos são iguais."),
    ])
    trechos = rb.buscar_trechos(col, EncoderFalso(), "saúde", n_final=10, esferas=None)
    assert [t["texto"] for t in trechos] == ["Art. 196. A saúde é direito de todos.", "Art. 5. Todos são iguais."]


def test_buscar_trechos_faz_busca_por_palavra_chave_e_numero_de_lei():
    col = ColecaoFalsa([doc("Art. 1. texto")])
    rb.buscar_trechos(col, EncoderFalso(), "cobrança Lei 8080", esferas=["federal"], palavras_chave=["gratuidade"])
    filtros_texto = [c["where_document"] for c in col.chamadas if c["where_document"]]
    assert filtros_texto == [{"$contains": "8.080"}, {"$contains": "gratuidade"}]
    assert {"esfera": {"$in": ["federal"]}} in [c["where"] for c in col.chamadas]


def test_buscar_trechos_ordena_pelo_reranker_e_aplica_nota_minima():
    col = ColecaoFalsa([doc("ruim"), doc("medio"), doc("bom")])
    reranker = RerankerFalso({"bom": 2.0, "medio": -4.0, "ruim": -8.0})
    trechos = rb.buscar_trechos(col, EncoderFalso(), "q", n_final=3, esferas=None, reranker=reranker, nota_minima=-5.0)
    assert [t["texto"] for t in trechos] == ["bom", "medio"]
    assert trechos[0]["nota"] == 2.0


def test_buscar_trechos_pode_voltar_vazio_quando_nada_passa_da_nota_minima():
    col = ColecaoFalsa([doc("irrelevante")])
    trechos = rb.buscar_trechos(col, EncoderFalso(), "q", esferas=None, reranker=RerankerFalso({}), nota_minima=-5.0)
    assert trechos == []
    assert rb.formatar_contexto(trechos) == "Nenhum artigo relevante foi encontrado na base local."


def test_buscar_trechos_usa_cache_para_consulta_repetida():
    col = ColecaoFalsa([doc("Art. 1. texto")])
    primeira = rb.buscar_trechos(col, EncoderFalso(), "q", esferas=None)
    chamadas = len(col.chamadas)
    segunda = rb.buscar_trechos(col, EncoderFalso(), "q", esferas=None)
    assert len(col.chamadas) == chamadas
    assert segunda == primeira
    segunda[0]["texto"] = "alterado"  # alterar o resultado não pode contaminar o cache
    assert rb.buscar_trechos(col, EncoderFalso(), "q", esferas=None)[0]["texto"] == "Art. 1. texto"


def test_formatar_contexto_mostra_uf_so_para_normas_estaduais():
    trechos = [
        {"texto": "A", "fonte": "ce-ba.md", "esfera": "estadual", "uf": "BA"},
        {"texto": "B", "fonte": "constituicao.md", "esfera": "constitucional", "uf": "BR"},
    ]
    contexto = rb.formatar_contexto(trechos)
    assert "(ce-ba.md, BA)" in contexto and "(constituicao.md)" in contexto


# ---------- verificar_citacoes / anotar_citacoes ----------

def test_verificar_citacoes_separa_o_que_veio_do_contexto():
    trechos = [{"texto": "**Art. 196.** A saúde é direito de todos. Lei nº 8.080, de 1990"}]
    resposta = "Conforme o art. 196 e o Art. 204, a Lei Orgânica (Lei nº 8.080/90) e a Lei 9.514/97."
    citacoes = rb.verificar_citacoes(resposta, trechos)
    assert citacoes == [
        {"citacao": "Art. 196", "no_contexto": True},
        {"citacao": "Art. 204", "no_contexto": False},
        {"citacao": "Lei 8.080", "no_contexto": True},
        {"citacao": "Lei 9.514", "no_contexto": False},
    ]


def test_anotar_citacoes_so_altera_quando_ha_citacao_fora():
    assert rb.anotar_citacoes("ok", [{"citacao": "Art. 196", "no_contexto": True}]) == "ok"
    anotada = rb.anotar_citacoes("parecer", [{"citacao": "Art. 230", "no_contexto": False}])
    assert anotada.startswith("parecer") and "Art. 230" in anotada and "NÃO aparecem" in anotada
