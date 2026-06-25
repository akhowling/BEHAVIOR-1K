import csv
import json
import re
from pathlib import Path

MANIFEST = Path("paper_outputs/log_manifest.csv")
OUT_DIR = Path("paper_outputs/mixed_replay_inputs")
OUT_CSV = Path("paper_outputs/replay_inputs.csv")

SAFE_PER_LOG = 1
UNSAFE_PER_LOG = 1
SAFETY_RADIUS = 0.75
SAFE_MARGIN = 0.03

OUT_DIR.mkdir(parents=True, exist_ok=True)

def slug(x):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(x))

def load_rows(path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def is_contact(r):
    return (r.get("num_contacts") or 0) > 0 or bool(r.get("contact_bodies_top5"))

def pick_evenly(items, n):
    if not items or n <= 0:
        return []
    if len(items) <= n:
        return items
    idxs = []
    for i in range(n):
        idxs.append(round(i * (len(items) - 1) / max(1, n - 1)))
    return [items[i] for i in sorted(set(idxs))]

manifest = list(csv.DictReader(open(MANIFEST)))
out_rows = []

for m in manifest:
    rows = load_rows(m["path"])
    steps = [r for r in rows if r.get("type") == "step"]

    if not steps:
        continue

    safe = [
        r for r in steps
        if not bool(r.get("safety_violation"))
        and not is_contact(r)
        and r.get("min_dist") is not None
        and float(r.get("min_dist")) > SAFETY_RADIUS + SAFE_MARGIN
    ]

    # Fallback to any non-violation step if margin is too strict.
    if not safe:
        safe = [
            r for r in steps
            if not bool(r.get("safety_violation"))
            and not is_contact(r)
            and r.get("min_dist") is not None
            and float(r.get("min_dist")) > SAFETY_RADIUS
        ]

    unsafe_all = [
        r for r in steps
        if bool(r.get("safety_violation")) or is_contact(r)
    ]

    # Prefer non-initial unsafe states when available, so replay captures
    # behavior-induced risk rather than only unsafe initialization.
    unsafe = [
        r for r in unsafe_all
        if (r.get("step") is not None and int(r.get("step")) >= 10)
    ]

    if not unsafe:
        unsafe = unsafe_all

    picks = []
    for r in pick_evenly(safe, SAFE_PER_LOG):
        picks.append(("safe", r))
    for r in pick_evenly(unsafe, UNSAFE_PER_LOG):
        picks.append(("unsafe", r))

    for label, step in picks:
        step_num = step.get("step")
        prefix_name = (
            f'{slug(m["task"])}__{slug(m["condition"])}__'
            f'{label}_step_{step_num}__{slug(Path(m["path"]).stem)}.jsonl'
        )
        prefix_path = OUT_DIR / prefix_name

        with prefix_path.open("w") as f:
            for row in rows:
                s = row.get("step")
                if s is None:
                    # Keep metadata/action rows without steps.
                    f.write(json.dumps(row) + "\n")
                elif s <= step_num:
                    f.write(json.dumps(row) + "\n")

        out_rows.append({
            "task": m["task"],
            "condition": m["condition"],
            "attack_type": m["attack_type"],
            "variant": m["variant"],
            "label": label,
            "source_log": m["path"],
            "prefix_log": str(prefix_path),
            "step": step_num,
            "min_dist": step.get("min_dist"),
            "safety_violation": bool(step.get("safety_violation")),
            "num_contacts": step.get("num_contacts"),
        })

with OUT_CSV.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "task", "condition", "attack_type", "variant", "label",
        "source_log", "prefix_log", "step", "min_dist",
        "safety_violation", "num_contacts"
    ])
    writer.writeheader()
    writer.writerows(out_rows)

print(f"Wrote {OUT_CSV} with {len(out_rows)} replay states")
for r in out_rows:
    print(r["task"], r["condition"], r["label"], "step", r["step"], "dist", r["min_dist"])
