from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows = []

    with Path(path).open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def load_scorer(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)

    if spec.loader is None:
        raise RuntimeError(f"Could not load scorer: {path}")

    spec.loader.exec_module(mod)
    return mod


def run_generated_vcio_scorers(
    log_path: str | Path,
    scorers_dir: str | Path = "TRACE/generated_vcio",
) -> Dict[str, Any]:
    rows = load_jsonl(log_path)
    scorer_paths = sorted(Path(scorers_dir).glob("*.py"))

    results = []

    for scorer_path in scorer_paths:
        if scorer_path.name == "__init__.py":
            continue

        mod = load_scorer(scorer_path)

        if not hasattr(mod, "score"):
            continue

        result = mod.score(rows)

        results.append({
            "scorer": str(scorer_path),
            "result": result,
        })

    scores = [
        r["result"]["score"]
        for r in results
        if isinstance(r.get("result"), dict)
        and isinstance(r["result"].get("score"), (int, float))
    ]

    return {
        "log_path": str(log_path),
        "scorers_dir": str(scorers_dir),
        "num_scorers": len(results),
        "scores": results,
        "aggregate": {
            "mean_score": sum(scores) / len(scores) if scores else None,
            "max_score": max(scores) if scores else None,
            "min_score": min(scores) if scores else None,
        },
    }
