import csv
import json
import re
from pathlib import Path

ROOTS = [
    Path("logs/pi05_eval_clean"),
    Path("logs/pi05_eval_action_noise"),
    Path("logs/pi05_eval_rgb_cutout"),
]

OUT = Path("paper_outputs/log_manifest.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

ACTION_VARIANTS = [
    "action_noise_low_seed_17",
    "action_noise_med_seed_17",
    "action_noise_high_seed_17",
]

RGB_VARIANTS = [
    "rgb_cutout_small_seed_17",
    "rgb_cutout_med_seed_17",
    "rgb_cutout_large_seed_17",
]

def load_steps(path):
    steps = []
    try:
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("type") == "step":
                    steps.append(r)
    except Exception:
        return []
    return steps

def find_variant_from_parts(parts, variants):
    joined = "/".join(parts)
    for v in variants:
        if v in parts:
            return v
        if f"rep_{v}" in joined:
            return v
        if re.search(re.escape(v) + r"[_/]", joined):
            return v
    return None

def infer_condition(path):
    parts = path.parts

    if "pi05_eval_clean" in parts:
        return "clean", "none", "clean"

    if "pi05_eval_action_noise" in parts:
        variant = find_variant_from_parts(parts, ACTION_VARIANTS)
        if variant is None:
            return "action_noise_legacy", "action_noise", "action_noise_legacy"
        return variant, "action_noise", variant.replace("_seed_17", "")

    if "pi05_eval_rgb_cutout" in parts:
        variant = find_variant_from_parts(parts, RGB_VARIANTS)
        if variant is None:
            return "rgb_cutout_legacy", "rgb_cutout", "rgb_cutout_legacy"
        return variant, "rgb_cutout", variant.replace("_seed_17", "")

    return "unknown", "unknown", "unknown"

rows = []

for root in ROOTS:
    if not root.exists():
        continue

    for path in sorted(root.rglob("safety_logger/*.jsonl")):
        s = str(path)

        if "offline_results" in s:
            continue
        if "tmp_prefix_logs" in s:
            continue
        if "pi05_eval_test" in s:
            continue
        if "old safety logger files" in s:
            continue
        if path.name.endswith("_events.txt"):
            continue

        task = path.name.split("_pi_eval_")[0]
        condition, attack_type, variant = infer_condition(path)
        steps = load_steps(path)

        # Drop failed/empty logs from paper tables.
        if len(steps) == 0:
            continue

        rows.append({
            "task": task,
            "condition": condition,
            "attack_type": attack_type,
            "variant": variant,
            "path": str(path),
            "num_steps": len(steps),
            "has_safe_steps": any(not bool(r.get("safety_violation")) for r in steps),
            "has_unsafe_steps": any(bool(r.get("safety_violation")) for r in steps),
        })

with OUT.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "task", "condition", "attack_type", "variant", "path",
        "num_steps", "has_safe_steps", "has_unsafe_steps"
    ])
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {OUT} with {len(rows)} nonempty logs")
for r in rows:
    print(f'{r["task"]:45s} {r["condition"]:30s} steps={r["num_steps"]}')
