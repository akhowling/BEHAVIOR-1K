import csv
from pathlib import Path
from collections import defaultdict

IN = Path("paper_outputs/safety_aggregate.csv")
OUT_CSV = Path("paper_outputs/raw_safety_rollout_table.csv")
OUT_TEX = Path("paper_outputs/raw_safety_rollout_table.tex")

def fnum(x):
    try:
        if x in ("", None, "None"):
            return None
        return float(x)
    except Exception:
        return None

def pct(x):
    return "--" if x is None else f"{100.0 * x:.1f}"

def num(x, nd=2):
    return "--" if x is None else f"{x:.{nd}f}"

def condition_family(condition):
    if condition == "clean":
        return "Clean"
    if condition.startswith("action_noise_") and "legacy" not in condition:
        return "Action noise"
    if condition.startswith("rgb_cutout_") and "legacy" not in condition:
        return "RGB cutout"
    return None

def short_task(task):
    names = {
        "cleaning_up_plates_and_food": "Cleaning plates/food",
        "make_microwave_popcorn": "Microwave popcorn",
        "picking_up_trash": "Picking up trash",
        "putting_away_Halloween_decorations": "Halloween decorations",
        "setting_mousetraps": "Setting mousetraps",
    }
    return names.get(task, task.replace("_", " "))

rows = list(csv.DictReader(open(IN)))

groups = defaultdict(list)
for r in rows:
    fam = condition_family(r["condition"])
    if fam is None:
        continue
    groups[(r["task"], fam)].append(r)

out = []

for (task, fam), vals in sorted(groups.items()):
    total_runs = sum(int(float(v["runs"])) for v in vals)

    def weighted_mean(field):
        xs = []
        ws = []
        for v in vals:
            x = fnum(v.get(field))
            w = int(float(v["runs"]))
            if x is not None:
                xs.append(x)
                ws.append(w)
        if not xs:
            return None
        return sum(x * w for x, w in zip(xs, ws)) / sum(ws)

    out.append({
        "task": task,
        "task_short": short_task(task),
        "condition_family": fam,
        "runs": total_runs,
        "violation_rate": weighted_mean("mean_violation_rate"),
        "contact_rate": weighted_mean("mean_contact_rate"),
        "mean_min_dist_m": weighted_mean("mean_min_dist_m"),
        "mean_collision_events": weighted_mean("mean_collision_events"),
    })

# Write CSV
with OUT_CSV.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "task", "task_short", "condition_family", "runs",
        "violation_rate", "contact_rate", "mean_min_dist_m",
        "mean_collision_events"
    ])
    writer.writeheader()
    writer.writerows(out)

# Write LaTeX table
lines = []
lines.append(r"\begin{table*}[t]")
lines.append(r"\centering")
lines.append(r"\small")
lines.append(r"\caption{Raw rollout safety metrics aggregated from structured safety traces. Action noise aggregates low, medium, and high action-noise settings; RGB cutout aggregates small, medium, and large cutout settings. Legacy runs are excluded from this table.}")
lines.append(r"\label{tab:raw-safety}")
lines.append(r"\begin{tabular}{llccccc}")
lines.append(r"\toprule")
lines.append(r"Task & Condition & Runs & Viol. Rate (\%) & Contact Rate (\%) & Mean $d_{\min}$ (m) & Collisions \\")
lines.append(r"\midrule")

for r in out:
    lines.append(
        f'{r["task_short"]} & {r["condition_family"]} & {r["runs"]} & '
        f'{pct(r["violation_rate"])} & {pct(r["contact_rate"])} & '
        f'{num(r["mean_min_dist_m"], 2)} & {num(r["mean_collision_events"], 1)} \\\\'
    )

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table*}")

OUT_TEX.write_text("\n".join(lines) + "\n")

print(f"Wrote {OUT_CSV}")
print(f"Wrote {OUT_TEX}")
print()
print(OUT_TEX.read_text())
