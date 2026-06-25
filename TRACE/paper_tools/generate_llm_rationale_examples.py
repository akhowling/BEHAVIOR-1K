import csv
import json
import os
import sys
import urllib.request
from pathlib import Path

INPUT = Path("paper_outputs/replay_inputs.csv")
OUT_CSV = Path("paper_outputs/llm_rationale_examples.csv")
OUT_TEX = Path("paper_outputs/llm_rationale_examples.tex")

PARCC_URL = os.environ.get("PARCC_URL", "https://litellm.parcc.upenn.edu").rstrip("/")
PARCC_MODEL = os.environ.get("PARCC_MODEL", "openai/gpt-oss-20b")
PARCC_API_KEY = os.environ.get("PARCC_API_KEY", "")

if not PARCC_API_KEY:
    raise SystemExit("Missing PARCC_API_KEY")

AUTH = PARCC_API_KEY if PARCC_API_KEY.startswith("Bearer ") else f"Bearer {PARCC_API_KEY}"

SAFETY_RADIUS = 0.75

def load_final_step(path):
    last = None
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("type") == "step":
                last = r
    return last or {}

def compact_obj(x):
    if x is None:
        return "unknown"
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        for k in ["name", "object", "label", "category", "body", "prim_path"]:
            if x.get(k):
                return str(x.get(k))
        return json.dumps(x)[:120]
    return str(x)[:120]

def pick_examples(rows):
    # Pick a compact set: one safe and one unsafe where possible, across different tasks.
    wanted = []
    used = set()

    # Prefer task diversity.
    for label in ["safe", "unsafe"]:
        for r in rows:
            key = (r["task"], label)
            if key in used:
                continue
            if r["label"] == label:
                wanted.append(r)
                used.add(key)
            if len(wanted) >= 8:
                return wanted

    return wanted

def call_llm(task, condition, label, step):
    min_dist = step.get("min_dist")
    violation = bool(step.get("safety_violation"))
    num_contacts = step.get("num_contacts", 0)
    closest_object = compact_obj(step.get("closest_object"))
    nearby_top = step.get("nearby_top", [])
    contact_bodies = step.get("contact_bodies_top5", [])

    if isinstance(nearby_top, list):
        nearby_summary = ", ".join(compact_obj(x) for x in nearby_top[:3])
    else:
        nearby_summary = compact_obj(nearby_top)

    if isinstance(contact_bodies, list):
        contact_summary = ", ".join(compact_obj(x) for x in contact_bodies[:3])
    else:
        contact_summary = compact_obj(contact_bodies)

    if label == "safe":
        action_catalog = (
            "a0 execute_policy_action allowed; "
            "a1 stop allowed but unnecessary unless risk is present; "
            "a2 back_up allowed; a3 turn_left allowed; a4 turn_right allowed"
        )
    else:
        action_catalog = (
            "a0 execute_policy_action blocked because the state is unsafe; "
            "a1 stop allowed; a2 back_up allowed; a3 turn_left allowed; a4 turn_right allowed"
        )

    user_prompt = f"""
Task: {task}
Condition: {condition}
Replay label: {label}

Structured safety evidence:
- min_dist: {min_dist}
- safety_radius: {SAFETY_RADIUS}
- safety_violation: {violation}
- num_contacts: {num_contacts}
- closest_object: {closest_object}
- nearby_objects: {nearby_summary}
- contact_bodies: {contact_summary}

Candidate actions:
{action_catalog}

Decision rules:
- If safety_violation=true, min_dist <= safety_radius, or contacts are present, choose stop unless a recovery action is clearly safer.
- If safety_violation=false, contacts=0, and min_dist > safety_radius, choose execute_policy_action to preserve task progress.
- Make the rationale task-specific. Mention the task and the relevant object or distance evidence.
- Do not claim task success. Do not invent objects not in the evidence.

Return valid JSON with:
chosen_action_id, confidence, safety_verdict, rationale, evidence_used, uncertainties.
"""

    payload = {
        "model": PARCC_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Return only valid JSON. No extra text. "
                    "safety_verdict must be exactly one of: safe, caution, unsafe."
                ),
            },
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
        "max_tokens": 260,
        "reasoning_effort": "low",
    }

    req = urllib.request.Request(
        f"{PARCC_URL}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": AUTH,
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    content = data["choices"][0]["message"]["content"]
    try:
        parsed = json.loads(content)
    except Exception:
        parsed = {"raw_content": content}

    return parsed

rows = list(csv.DictReader(open(INPUT)))
examples = pick_examples(rows)

out = []

for r in examples:
    step = load_final_step(r["prefix_log"])
    print("Calling LLM:", r["task"], r["condition"], r["label"], flush=True)

    parsed = call_llm(r["task"], r["condition"], r["label"], step)

    chosen_id = parsed.get("chosen_action_id", "")
    chosen_action = {
        "a0": "execute_policy_action",
        "a1": "stop",
        "a2": "back_up",
        "a3": "turn_left",
        "a4": "turn_right",
    }.get(chosen_id, chosen_id)

    out.append({
        "task": r["task"],
        "condition": r["condition"],
        "label": r["label"],
        "min_dist": step.get("min_dist"),
        "safety_violation": step.get("safety_violation"),
        "num_contacts": step.get("num_contacts"),
        "closest_object": compact_obj(step.get("closest_object")),
        "chosen_action_id": chosen_id,
        "chosen_action": chosen_action,
        "confidence": parsed.get("confidence", ""),
        "safety_verdict": parsed.get("safety_verdict", ""),
        "rationale": parsed.get("rationale", parsed.get("raw_content", "")),
        "evidence_used": parsed.get("evidence_used", ""),
        "uncertainties": parsed.get("uncertainties", ""),
    })

with OUT_CSV.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(out[0].keys()))
    writer.writeheader()
    writer.writerows(out)

# Make a compact LaTeX table with 4 examples max.
def esc(s):
    s = str(s)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for a, b in repl.items():
        s = s.replace(a, b)
    return s

shown = out[:4]
lines = []
lines.append(r"\begin{table}[t]")
lines.append(r"\centering")
lines.append(r"\small")
lines.append(r"\caption{Representative task-specific LLM commander rationales. The model receives structured trace evidence and outputs a JSON action ID with an auditable rationale. Only the validated action ID is used by the system.}")
lines.append(r"\label{tab:llm-rationales}")
lines.append(r"\begin{tabular}{lllp{4.5cm}}")
lines.append(r"\toprule")
lines.append(r"Task & State & Action & Rationale \\")
lines.append(r"\midrule")
for r in shown:
    lines.append(
        f'{esc(r["task"].replace("_", " "))} & '
        f'{esc(r["label"])} & '
        f'\\texttt{{{esc(r["chosen_action"])}}} & '
        f'{esc(r["rationale"])} \\\\'
    )
lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")

OUT_TEX.write_text("\n".join(lines) + "\n")

print(f"Wrote {OUT_CSV}")
print(f"Wrote {OUT_TEX}")
print()
print(OUT_TEX.read_text())
