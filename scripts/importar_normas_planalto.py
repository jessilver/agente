"""Baixa normas federais de saúde do planalto.gov.br e grava em Markdown no formato do indexador_rag.py.

- Remove o texto revogado/alterado (tachado no site: <strike>, <s>, <del>).
- Agrupa cada artigo (caput, incisos, parágrafos) num bloco; artigos longos viram blocos de até
  TAMANHO_BLOCO caracteres que repetem o rótulo "Art. Nº (continuação)", para cada trecho indexado
  saber de qual artigo é.
- Gera frontmatter YAML (titulo, tipo, numero, data, ementa, fonte_url) lido pelo indexador.

Uso:
    python scripts/importar_normas_planalto.py                    # baixa do Planalto
    python scripts/importar_normas_planalto.py --html PASTA       # usa HTMLs já baixados (mesmo nome do link)
Depois, indexe só esta pasta (não toca no restante da base):
    python indexador_rag.py --base data/normas_federais_saude
"""

import argparse
import html
import re
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
PASTA_SAIDA = RAIZ / "data" / "normas_federais_saude" / "federais" / "saude"
TAMANHO_BLOCO = 480  # o indexador corta parágrafos acima de 500 caracteres

NORMAS = [
    {
        "url": "https://www.planalto.gov.br/ccivil_03/leis/l8080.htm",
        "arquivo": "lei-8080-1990.md",
        "titulo": "Lei nº 8.080, de 19 de setembro de 1990 (Lei Orgânica da Saúde)",
        "tipo": "Lei Ordinária",
        "numero": "8080",
        "data": "1990-09-19",
    },
    {
        "url": "https://www.planalto.gov.br/ccivil_03/leis/l8142.htm",
        "arquivo": "lei-8142-1990.md",
        "titulo": "Lei nº 8.142, de 28 de dezembro de 1990",
        "tipo": "Lei Ordinária",
        "numero": "8142",
        "data": "1990-12-28",
    },
    {
        "url": "https://www.planalto.gov.br/ccivil_03/leis/lcp/lcp141.htm",
        "arquivo": "lei-complementar-141-2012.md",
        "titulo": "Lei Complementar nº 141, de 13 de janeiro de 2012",
        "tipo": "Lei Complementar",
        "numero": "141",
        "data": "2012-01-13",
    },
    {
        "url": "https://www.planalto.gov.br/ccivil_03/_ato2011-2014/2011/decreto/d7508.htm",
        "arquivo": "decreto-7508-2011.md",
        "titulo": "Decreto nº 7.508, de 28 de junho de 2011",
        "tipo": "Decreto",
        "numero": "7508",
        "data": "2011-06-28",
    },
]

_TAGS_REVOGADAS = {"strike", "s", "del"}
_TAGS_BLOCO = {"p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "blockquote", "table"}
_RE_ARTIGO = re.compile(r"^Art\.\s*(\d+[º°o]?(?:-[A-Z])?)\.?", re.I)
_RE_ESPACOS = re.compile(r"[ \t ]+")


class _ExtratorTexto(HTMLParser):
    """Extrai as linhas de texto do HTML, ignorando trechos tachados (revogados), scripts e estilos."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.linhas = [""]
        self.ignorando = 0

    def handle_starttag(self, tag, attrs):
        if tag in _TAGS_REVOGADAS or tag in ("script", "style"):
            self.ignorando += 1
        elif tag in _TAGS_BLOCO:
            self.linhas.append("")

    def handle_endtag(self, tag):
        if tag in _TAGS_REVOGADAS or tag in ("script", "style"):
            self.ignorando = max(0, self.ignorando - 1)
        elif tag in _TAGS_BLOCO:
            self.linhas.append("")

    def handle_data(self, data):
        if not self.ignorando:
            self.linhas[-1] += data


def extrair_linhas(conteudo_html: str) -> list[str]:
    parser = _ExtratorTexto()
    parser.feed(conteudo_html)
    linhas = []
    for linha in parser.linhas:
        linha = _RE_ESPACOS.sub(" ", html.unescape(linha).replace("\r", " ").replace("\n", " ")).strip()
        if linha:
            linhas.append(linha)
    return linhas


def separar_cabecalho(linhas: list[str]) -> tuple[str, list[str]]:
    """Devolve (ementa, linhas do texto normativo a partir do Art. 1º)."""
    inicio = next((i for i, l in enumerate(linhas) if _RE_ARTIGO.match(l)), 0)
    cabecalho = linhas[:inicio]
    # A ementa é o parágrafo mais longo antes do Art. 1º que não seja o preâmbulo ("O PRESIDENTE...").
    candidatas = [l for l in cabecalho if len(l) > 40 and not l.upper().startswith(("O PRESIDENTE", "A PRESIDENTA"))
                  and "Presidência da República" not in l]
    ementa = max(candidatas, key=len) if candidatas else ""
    return ementa, linhas[inicio:]


def agrupar_artigos(linhas: list[str]) -> list[str]:
    """Junta cada artigo num bloco e quebra os longos em blocos de até TAMANHO_BLOCO com o rótulo repetido."""
    artigos, atual, rotulo = [], [], None
    for linha in linhas:
        m = _RE_ARTIGO.match(linha)
        if m:
            if atual:
                artigos.append((rotulo, atual))
            atual, rotulo = [linha], f"Art. {m.group(1)}"
        elif rotulo:
            atual.append(linha)
    if atual:
        artigos.append((rotulo, atual))

    blocos = []
    for rotulo, partes in artigos:
        bloco = ""
        for parte in partes:
            # Linhas maiores que o bloco são fatiadas para não serem cortadas sem rótulo pelo indexador.
            pedacos = [parte[i: i + TAMANHO_BLOCO - 40] for i in range(0, len(parte), TAMANHO_BLOCO - 40)]
            for pedaco in pedacos:
                candidato = f"{bloco}\n{pedaco}" if bloco else pedaco
                if len(candidato) <= TAMANHO_BLOCO:
                    bloco = candidato
                else:
                    blocos.append(bloco)
                    bloco = f"{rotulo} (continuação)\n{pedaco}"
        if bloco:
            blocos.append(bloco)
    return blocos


def _yaml(valor: str) -> str:
    return '"' + valor.replace("\\", "\\\\").replace('"', '\\"') + '"'


def converter(norma: dict, conteudo_html: str) -> str:
    ementa, linhas = separar_cabecalho(extrair_linhas(conteudo_html))
    blocos = agrupar_artigos(linhas)
    frontmatter = "\n".join([
        "---",
        f"titulo: {_yaml(norma['titulo'])}",
        f"tipo: {_yaml(norma['tipo'])}",
        f"numero: {_yaml(norma['numero'])}",
        f"data: {_yaml(norma['data'])}",
        f"ementa: {_yaml(ementa)}",
        f"fonte_url: {_yaml(norma['url'])}",
        "---",
    ])
    return frontmatter + "\n\n" + "\n\n".join(blocos) + "\n"


def baixar(url: str) -> str:
    pedido = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; agente-sus-indexador)"})
    with urllib.request.urlopen(pedido, timeout=60) as resposta:
        return _decodificar(resposta.read())


def _decodificar(bruto: bytes) -> str:
    try:
        return bruto.decode("utf-8")
    except UnicodeDecodeError:
        return bruto.decode("cp1252", errors="replace")  # padrão das páginas antigas do Planalto


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--html", type=Path, help="Pasta com os HTMLs já baixados (nome igual ao do link).")
    parser.add_argument("--saida", type=Path, default=PASTA_SAIDA)
    args = parser.parse_args()

    args.saida.mkdir(parents=True, exist_ok=True)
    for norma in NORMAS:
        if args.html:
            conteudo = _decodificar((args.html / norma["url"].rsplit("/", 1)[-1]).read_bytes())
        else:
            conteudo = baixar(norma["url"])
        markdown = converter(norma, conteudo)
        destino = args.saida / norma["arquivo"]
        destino.write_text(markdown, encoding="utf-8")
        n_blocos = markdown.count("\n\n")
        print(f"{destino.relative_to(RAIZ)}: {n_blocos} blocos, {len(markdown)} caracteres")


if __name__ == "__main__":
    main()
