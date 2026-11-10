"""Testes do snapshots.py gravando numa pasta temporária."""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import snapshots  # noqa: E402


def test_salvar_snapshot_grava_json_e_markdown(tmp_path, monkeypatch):
    monkeypatch.setattr(snapshots, "PASTA_SNAPSHOTS", str(tmp_path))
    trechos = [{"texto": "Art. 196", "fonte": "constituicao.md", "esfera": "constitucional", "uf": "BR",
                "nota": -3.74, "origem": "vetorial"}]
    citacoes = [{"citacao": "Art. 196", "no_contexto": True}, {"citacao": "Art. 230", "no_contexto": False}]

    caminho_md = snapshots.salvar_snapshot(
        "teste", "cpu", "consulta", trechos, "resposta", {"busca_rag": 1.5}, entrada="entrada",
        extras={"citacoes": citacoes},
    )

    with open(caminho_md, encoding="utf-8") as f:
        md = f.read()
    with open(caminho_md[:-3] + ".json", encoding="utf-8") as f:
        dados = json.load(f)

    assert dados["dispositivo"] == "cpu" and dados["trechos"] == trechos and dados["citacoes"] == citacoes
    assert "gpu" not in dados["hardware"]  # execução em CPU não registra GPU/VRAM
    assert "constituicao.md (constitucional · BR · nota -3.74 · via vetorial)" in md
    assert "Citações confirmadas nos trechos:** Art. 196" in md
    assert "Citações fora dos trechos:** Art. 230" in md


def test_cronometro_registra_cada_etapa():
    cron = snapshots.Cronometro()
    with cron.etapa("a"):
        pass
    with cron.etapa("b"):
        pass
    assert set(cron.etapas) == {"a", "b"} and all(v >= 0 for v in cron.etapas.values())
