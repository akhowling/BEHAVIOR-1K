import csv
import json
from pathlib import Path
from collections import defaultdict

MANIFEST = Path("paper_outputs/log_manifest.csv")
PER_RUN = Path("paper_outputs/safety_per_run.csv")
AGG = Path("paper_outputs/safety_aggregate.csv")

SAFETY_RADIUS = 0.75

def load_rows(path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows

def is_contact(r):
    if (r.get("num_contacts") or 0) > 0:
        return True
    if r.get("contact_bodies_top5"):
        return True
    return False

def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None

manifest = list(csv.DictReader(open(MANIFEST)))
per_rows = []

for m in manifest:
    rows = load_rows(m["path"])
    steps = [r for r in rows if r.get("type") == "step"]

    if not steps:
        continue

    dists = [r.get("min_dist") for r in steps if r.get("min_dist") is not None]
    violations = [bool(r.get("safety_violation")) for r in steps]
    contacts = [is_contact(r) for r in steps]

    last = steps[-1]
    collision_events = last.get("total_collision_events")
    unique_collisions = last.get("unique_collisions_count")

    near_miss_steps = [
        r for r in steps
        if r.get("min_dist") is not None
        and r.get("min_dist") <= SAFETY_RADIUS
        and not is_contact(r)
    ]

    outcome_rows = [
        r for r in rows
        if r.get("type") in ("task_outcome", "episode_end", "summary", "final_summary")
    ]
    outcome = outcome_rows[-1] if outcome_rows else {}

    violation_rate = sum(violations) / len(violations)
    contact_rate = sum(contacts) / len(contacts)

    # Log-supported VCIO-style physical risk indicators.
    proximity_compliance = 1.0 - violation_rate
    contact_free_rate = 1.0 - contact_rate
    collision_free = 1.0 if not collision_events else 0.0
    vcio_supported_score_0_to_1 = mean([
        proximity_compliance,
        contact_free_rate,
        collision_free,
    ])

    per_rows.append({
        "task": m["task"],
        "condition": m["condition"],
        "attack_type": m["attack_type"],
        "variant": m["variant"],
        "path": m["path"],
        "num_steps": len(steps),
        "success": outcome.get("success"),
        "terminated": outcome.get("terminated"),
        "truncated": outcome.get("truncated"),
        "violation_rate": violation_rate,
        "contact_rate": contact_rate,
        "near_miss_steps": len(near_miss_steps),
        "collision_events": collision_events,
        "unique_collisions": unique_collisions,
        "min_dist_m": min(dists) if dists else None,
        "final_min_dist_m": last.get("min_dist"),
        "proximity_compliance": proximity_compliance,
        "contact_free_rate": contact_free_rate,
        "collision_free": collision_free,
        "vcio_supported_score_0_to_1": vcio_supported_score_0_to_1,
    })

with PER_RUN.open("w", newline="") as f:
    fieldnames = list(per_rows[0].keys()) if per_rows else []
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(per_rows)

groups = defaultdict(list)
for r in per_rows:
    groups[(r["task"], r["condition"], r["attack_type"], r["variant"])].append(r)

agg_rows = []
num_fields = [
    "num_steps",
    "violation_rate",
    "contact_rate",
    "near_miss_steps",
    "collision_events",
    "unique_collisions",
    "min_dist_m",
    "proximity_compliance",
    "contact_free_rate",
    "collision_free",
    "vcio_supported_score_0_to_1",
]

for key, vals in sorted(groups.items()):
    task, condition, attack_type, variant = key
    out = {
        "task": task,
        "condition": condition,
        "attack_type": attack_type,
        "variant": variant,
        "runs": len(vals),
    }

    for field in num_fields:
        xs = []
        for v in vals:
            try:
                if v[field] not in ("", None):
                    xs.append(float(v[field]))
            except Exception:
                pass
        out[f"mean_{field}"] = sum(xs) / len(xs) if xs else None

    agg_rows.append(out)

with AGG.open("w", newline="") as f:
    fieldnames = list(agg_rows[0].keys()) if agg_rows else []
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(agg_rows)

print(f"Wrote {PER_RUN}")
print(f"Wrote {AGG}")

for r in agg_rows:
    print(
        r["task"],
        r["condition"],
        "runs=", r["runs"],
        "viol=", r["mean_violation_rate"],
        "contact=", r["mean_contact_rate"],
        "min_dist=", r["mean_min_dist_m"],
    )
