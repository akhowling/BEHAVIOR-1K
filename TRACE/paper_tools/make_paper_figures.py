import csv
import json
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt

OUT = Path("paper_outputs/figures")
OUT.mkdir(parents=True, exist_ok=True)

SAFETY_RADIUS = 0.75

def load_steps(path):
    steps = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("type") == "step":
                steps.append(r)
    return steps

# ------------------------------------------------------------
# Figure 1: temporal failure trace
# ------------------------------------------------------------
manifest = list(csv.DictReader(open("paper_outputs/log_manifest.csv")))

target = None
for r in manifest:
    if r.get("task") == "make_microwave_popcorn" and r.get("condition") == "clean":
        target = r
        break

if target is None:
    # Fallback: choose the run with most collisions in safety_per_run
    runs = list(csv.DictReader(open("paper_outputs/safety_per_run.csv")))
    runs.sort(key=lambda x: float(x.get("collision_events", 0) or 0), reverse=True)
    target = runs[0]

log_path = target.get("log_path") or target.get("path") or target.get("file") or target.get("prefix_log")
if not log_path:
    # Try common field names from manifest
    print("Could not find log path fields in target:", target.keys())
else:
    steps = load_steps(log_path)
    xs = []
    ds = []
    contacts_x = []
    contacts_y = []
    for i, s in enumerate(steps):
        d = s.get("min_dist")
        if d is None:
            d = s.get("min_center_dist")
        if d is None:
            continue
        try:
            d = float(d)
        except Exception:
            continue
        xs.append(i)
        ds.append(d)
        if int(s.get("num_contacts", 0) or 0) > 0:
            contacts_x.append(i)
            contacts_y.append(d)

    plt.figure(figsize=(7.0, 3.0))
    plt.plot(xs, ds, linewidth=1.5, label=r"$d_{\min}(t)$")
    plt.axhline(SAFETY_RADIUS, linestyle="--", linewidth=1.2, label=r"$r_{\mathrm{safety}}$")
    if contacts_x:
        plt.scatter(contacts_x, contacts_y, s=12, marker="x", label="contact")
    plt.xlabel("Timestep")
    plt.ylabel("Minimum distance (m)")
    plt.title("Example temporal safety trace")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / "failure_trace_distance.pdf")
    plt.savefig(OUT / "failure_trace_distance.png", dpi=250)
    plt.close()

# ------------------------------------------------------------
# Figure 2: violation/contact rates by task-condition
# ------------------------------------------------------------
rows = list(csv.DictReader(open("paper_outputs/safety_aggregate.csv")))

# If aggregate is too detailed, make labels compact and skip legacy
plot_rows = []
for r in rows:
    cond = r.get("condition", "")
    if "legacy" in cond.lower():
        continue
    task = r.get("task", "")
    if not task:
        continue
    try:
        viol = float(r.get("violation_rate", r.get("mean_violation_rate", 0)) or 0) * 100
        contact = float(r.get("contact_rate", r.get("mean_contact_rate", 0)) or 0) * 100
    except Exception:
        continue
    label = task.replace("cleaning_up_plates_and_food", "plates").replace("make_microwave_popcorn", "popcorn").replace("picking_up_trash", "trash").replace("putting_away_Halloween_decorations", "halloween").replace("setting_mousetraps", "mousetraps")
    label = label + "\\n" + cond.replace("_seed_17", "").replace("action_noise", "noise").replace("rgb_cutout", "cutout")
    plot_rows.append((label, viol, contact))

# Limit to most readable rows if needed
if len(plot_rows) > 14:
    # Prefer rows with highest violation/contact plus some safe rows
    plot_rows = sorted(plot_rows, key=lambda x: x[1] + x[2], reverse=True)[:14]

labels = [x[0] for x in plot_rows]
viol = [x[1] for x in plot_rows]
contact = [x[2] for x in plot_rows]

x = range(len(labels))
width = 0.4

plt.figure(figsize=(8.0, 3.6))
plt.bar([i - width/2 for i in x], viol, width, label="Violation rate")
plt.bar([i + width/2 for i in x], contact, width, label="Contact rate")
plt.ylabel("Rate (%)")
plt.xticks(list(x), labels, rotation=45, ha="right", fontsize=7)
plt.legend(fontsize=8)
plt.title("Safety-trace failure rates")
plt.tight_layout()
plt.savefig(OUT / "violation_contact_rates.pdf")
plt.savefig(OUT / "violation_contact_rates.png", dpi=250)
plt.close()

# ------------------------------------------------------------
# Figure 3: commander replay action choices
# ------------------------------------------------------------
rows = list(csv.DictReader(open("paper_outputs/commander_comparison.csv")))

counts = {
    "safe_policy": 0,
    "safe_stop": 0,
    "unsafe_policy": 0,
    "unsafe_stop": 0,
}
for r in rows:
    label = r.get("label")
    choice = r.get("llm_choice", "")
    if label == "safe":
        if choice == "execute_policy_action":
            counts["safe_policy"] += 1
        elif choice == "stop":
            counts["safe_stop"] += 1
    elif label == "unsafe":
        if choice == "execute_policy_action":
            counts["unsafe_policy"] += 1
        elif choice == "stop":
            counts["unsafe_stop"] += 1

plt.figure(figsize=(4.8, 3.2))
groups = ["Safe states", "Unsafe states"]
policy_vals = [counts["safe_policy"], counts["unsafe_policy"]]
stop_vals = [counts["safe_stop"], counts["unsafe_stop"]]
x = range(len(groups))
plt.bar(x, policy_vals, label="execute_policy_action")
plt.bar(x, stop_vals, bottom=policy_vals, label="stop")
plt.xticks(list(x), groups)
plt.ylabel("Replay states")
plt.title("Validated commander replay choices")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUT / "commander_replay_choices.pdf")
plt.savefig(OUT / "commander_replay_choices.png", dpi=250)
plt.close()

print("Wrote figures to", OUT)
for p in sorted(OUT.iterdir()):
    print(p)
