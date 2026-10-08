import json
from pathlib import Path
from typing import Any

from datasets import Dataset


def load_vigilancia_sanitaria_dataset(path: str | Path) -> Dataset:
    """Carrega o dataset bruto e o converte para o formato de texto do SFT."""
    dataset_path = Path(path)
    records: list[dict[str, Any]] = []

    with dataset_path.open("r", encoding="utf-8") as dataset_file:
        for line_number, line in enumerate(dataset_file, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("//"):
                continue

            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSON inválido em {dataset_path} na linha {line_number}: {exc.msg}"
                ) from exc

            if not isinstance(row, dict) or not all(
                field in row for field in ("instruction", "input", "output")
            ):
                raise ValueError(
                    f"Registro inválido em {dataset_path} na linha {line_number}: "
                    "esperado instruction, input e output"
                )

            instruction = str(row["instruction"]).strip()
            input_text = str(row["input"] or "").strip()
            output_text = str(row["output"] or "").strip()
            if not instruction or not output_text:
                raise ValueError(
                    f"Registro inválido em {dataset_path} na linha {line_number}: "
                    "instrução e resposta são obrigatórias"
                )

            question = f"### Pergunta: {instruction}"
            if input_text:
                question += f"\n\n{input_text}"

            records.append(
                {
                    "text": f"{question}\n\n### Resposta: {output_text}",
                }
            )

    if not records:
        raise ValueError(f"Nenhum registro válido encontrado em {dataset_path}")

    return Dataset.from_list(records)
