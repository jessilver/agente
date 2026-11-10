"""Registro de execuções do agente em snapshots/ (JSON + Markdown), para comparar versões e montar apresentações."""

import json
import os
import platform
import subprocess
import time
from datetime import datetime

DIRETORIO_RAIZ = os.path.dirname(os.path.abspath(__file__))
PASTA_SNAPSHOTS = os.path.join(DIRETORIO_RAIZ, "snapshots")


class Cronometro:
    """Mede a duração de cada etapa: with cron.etapa("busca"): ..."""

    def __init__(self):
        self.etapas = {}

    def etapa(self, nome):
        cron = self

        class _Etapa:
            def __enter__(self):
                self.inicio = time.perf_counter()

            def __exit__(self, *exc):
                cron.etapas[nome] = round(time.perf_counter() - self.inicio, 2)

        return _Etapa()


def _git(*args):
    return subprocess.run(["git", "-C", DIRETORIO_RAIZ, *args], capture_output=True, text=True).stdout.strip()


def _commit_git():
    try:
        return {
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "commit": _git("rev-parse", "--short", "HEAD"),
            "alteracoes_locais": bool(_git("status", "--porcelain", "--untracked-files=no")),
        }
    except Exception:
        return {"branch": "", "commit": "", "alteracoes_locais": False}


def _hardware(dispositivo):
    info = {"sistema": platform.platform(), "processador": platform.processor()}
    try:
        import torch

        info["torch"] = torch.__version__
        if str(dispositivo).startswith("cuda") and torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["vram_alocada_gb"] = round(torch.cuda.memory_allocated() / 1024 ** 3, 2)
            info["vram_pico_gb"] = round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)
    except Exception:
        pass
    return info


def salvar_snapshot(script, dispositivo, consulta, trechos, resposta, tempos, entrada="", extras=None):
    """Grava snapshots/<data>_<script>.json e .md e devolve o caminho do .md."""
    os.makedirs(PASTA_SNAPSHOTS, exist_ok=True)
    agora = datetime.now()
    base = os.path.join(PASTA_SNAPSHOTS, f"{agora:%Y-%m-%d_%H%M%S}_{script}")

    dados = {
        "data": agora.isoformat(timespec="seconds"),
        "script": script,
        "dispositivo": dispositivo,
        "git": _commit_git(),
        "hardware": _hardware(dispositivo),
        "tempos_s": tempos,
        "consulta_rag": consulta,
        "entrada": entrada,
        "trechos": trechos,
        "resposta": resposta,
        **(extras or {}),
    }
    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump(dados, f, ensure_ascii=False, indent=2)

    linhas = [
        f"# {script} — {agora:%d/%m/%Y %H:%M:%S}",
        "",
        f"- **Dispositivo:** {dispositivo} ({dados['hardware'].get('gpu', dados['hardware'].get('processador', ''))})",
        f"- **Git:** `{dados['git']['branch']}` @ `{dados['git']['commit']}`"
        + (" (com alterações não commitadas)" if dados["git"]["alteracoes_locais"] else ""),
        f"- **Tempos (s):** " + ", ".join(f"{k} {v}" for k, v in tempos.items()),
    ]
    if "vram_pico_gb" in dados["hardware"]:
        linhas.append(f"- **VRAM (pico):** {dados['hardware']['vram_pico_gb']} GB")
    for chave, valor in (extras or {}).items():
        linhas.append(f"- **{chave}:** {valor}")
    linhas += ["", "## Consulta RAG", "", consulta, "", "## Trechos recuperados", ""]
    for i, t in enumerate(trechos, 1):
        nota = f" · nota {t['nota']:.2f}" if t.get("nota") is not None else ""
        linhas += [f"### {i}. {t['fonte']} ({t['esfera']}{nota})", "", t["texto"], ""]
    if entrada:
        linhas += ["## Entrada", "", entrada, ""]
    linhas += ["## Resposta", "", resposta, ""]

    with open(base + ".md", "w", encoding="utf-8") as f:
        f.write("\n".join(linhas))
    return base + ".md"
