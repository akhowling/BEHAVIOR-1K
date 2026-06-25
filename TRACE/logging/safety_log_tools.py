import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.tools import tool


# ============================================================
# Basic JSONL loading helpers
# ============================================================

def _load_jsonl(log_path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(Path(log_path).expanduser(), "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    rows.append({
                        "type": "parse_error",
                        "raw_line": line,
                    })
    return rows


def _step_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [r for r in rows if r.get("type") == "step"]


def _latest_step(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    steps = _step_rows(rows)
    return steps[-1] if steps else None


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def _dist3(a: List[float], b: List[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    if len(a) < 2 or len(b) < 2:
        return None

    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])

    az = float(a[2]) if len(a) > 2 else 0.0
    bz = float(b[2]) if len(b) > 2 else 0.0

    dx = ax - bx
    dy = ay - by
    dz = az - bz

    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _get_nearby_list(step: Dict[str, Any]) -> List[Dict[str, Any]]:
    nearby = step.get("nearby_top", []) or []
    out = []

    for item in nearby:
        if isinstance(item, dict):
            name = item.get("name")
            dist = _safe_float(item.get("dist"))
            pos = item.get("pos") or item.get("position")
            out.append({
                "name": name,
                "dist": dist,
                "pos": pos,
                "raw": item,
            })
        else:
            out.append({
                "name": str(item),
                "dist": None,
                "pos": None,
                "raw": item,
            })

    out = [x for x in out if x.get("name") is not None]
    out.sort(key=lambda x: x["dist"] if x["dist"] is not None else float("inf"))
    return out


def _compute_velocity_samples(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    steps = _step_rows(rows)
    speeds = []
    prev = None

    for r in steps:
        pos = r.get("robot_pos")
        t = _safe_float(r.get("t_wall"))

        if pos is None or t is None:
            continue

        if prev is not None:
            p0, t0 = prev
            dt = t - t0

            if dt > 1e-6:
                d = _dist3(pos, p0)

                if d is not None:
                    speeds.append({
                        "step": r.get("step"),
                        "t_wall": t,
                        "speed_mps": d / dt,
                        "robot_pos": pos,
                    })

        prev = (pos, t)

    return speeds


def _latest_speed(rows: List[Dict[str, Any]]) -> Optional[float]:
    samples = _compute_velocity_samples(rows)
    if not samples:
        return None
    return samples[-1].get("speed_mps")


def _recent_steps(rows: List[Dict[str, Any]], last_n_steps: int = 20) -> List[Dict[str, Any]]:
    steps = _step_rows(rows)
    return steps[-last_n_steps:] if steps else []


def _is_contact_step(r: Dict[str, Any]) -> bool:
    if _safe_int(r.get("num_contacts")) and _safe_int(r.get("num_contacts")) > 0:
        return True

    contacts = r.get("contact_bodies_top5", []) or []
    return len(contacts) > 0


def _normalize_action_text(action: Any) -> str:
    if isinstance(action, str):
        return action.lower().strip()

    if isinstance(action, dict):
        parts = []
        for k, v in action.items():
            parts.append(f"{k}:{v}")
        return " ".join(parts).lower().strip()

    if isinstance(action, list) or isinstance(action, tuple):
        return " ".join([str(x) for x in action]).lower().strip()

    return str(action).lower().strip()


def _classify_action(action: Any) -> Dict[str, Any]:
    """
    Best effort action classifier.
    It handles strings, dictionaries, and simple numeric action vectors.
    """

    text = _normalize_action_text(action)
    if isinstance(action, dict):
        name = str(action.get("name", "")).lower()
        kind = str(action.get("kind", "")).lower()

        if name == "execute_policy_action" or kind == "raw_policy_action":
            return {
                "action_text": text,
                "category": "policy",
                "numeric_hint": None,
            }

    stop_words = [
        "stop",
        "halt",
        "hold",
        "wait",
        "noop",
        "no_op",
        "no op",
        "brake",
        "emergency_stop",
        "emergency stop",
        "freeze",
    ]

    forward_words = [
        "forward",
        "move_forward",
        "move forward",
        "advance",
        "ahead",
        "approach",
        "go_to",
        "goto",
        "navigate",
        "nav_to",
    ]

    back_words = [
        "back",
        "backward",
        "reverse",
        "retreat",
        "move_back",
        "move back",
    ]

    turn_words = [
        "turn",
        "rotate",
        "yaw",
        "left",
        "right",
        "spin",
    ]

    manip_words = [
        "grasp",
        "grab",
        "pick",
        "place",
        "open",
        "close",
        "push",
        "pull",
        "toggle",
        "press",
        "lift",
        "drop",
        "release",
        "slice",
        "soak",
        "clean",
        "wipe",
        "pour",
    ]

    category = "unknown"

    if any(w in text for w in stop_words):
        category = "stop"
    elif any(w in text for w in back_words):
        category = "back"
    elif any(w in text for w in forward_words):
        category = "forward"
    elif any(w in text for w in turn_words):
        category = "turn"
    elif any(w in text for w in manip_words):
        category = "manipulation"

    numeric_hint = None

    if isinstance(action, list) or isinstance(action, tuple):
        vals = []
        for x in action:
            try:
                vals.append(float(x))
            except Exception:
                pass

        if vals:
            numeric_hint = vals

            # Common mobile robot convention:
            # [linear_velocity, angular_velocity] or [vx, vy, wz]
            if len(vals) >= 1:
                if abs(vals[0]) < 1e-6 and all(abs(v) < 1e-6 for v in vals):
                    category = "stop"
                elif vals[0] > 0:
                    category = "forward"
                elif vals[0] < 0:
                    category = "back"
                elif len(vals) >= 2 and abs(vals[1]) > 1e-6:
                    category = "turn"

    return {
        "action_text": text,
        "category": category,
        "numeric_hint": numeric_hint,
    }


def _extract_available_actions_from_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Looks for available action fields in recent rows.
    This avoids making the LLM invent available actions.
    """

    candidates = []

    for r in reversed(rows):
        for key in [
            "available_actions",
            "candidate_actions",
            "actions",
            "action_options",
            "valid_actions",
        ]:
            if key in r and r.get(key) is not None:
                val = r.get(key)
                if isinstance(val, list):
                    candidates = val
                else:
                    candidates = [val]

                return {
                    "found": True,
                    "source_type": r.get("type"),
                    "source_step": r.get("step"),
                    "source_key": key,
                    "actions": candidates,
                }

    return {
        "found": False,
        "source_type": None,
        "source_step": None,
        "source_key": None,
        "actions": [],
    }


def _risk_from_state_and_action(
    latest: Dict[str, Any],
    rows: List[Dict[str, Any]],
    action: Any,
    safety_radius_m: float,
    hard_stop_radius_m: float,
    speed_limit_mps: float,
) -> Dict[str, Any]:

    info = _classify_action(action)
    category = info["category"]

    min_dist = _safe_float(latest.get("min_dist"))
    closest_object = latest.get("closest_object")
    num_contacts = _safe_int(latest.get("num_contacts")) or 0
    contact_bodies = latest.get("contact_bodies_top5", []) or []
    had_contact = num_contacts > 0 or len(contact_bodies) > 0
    safety_violation = bool(latest.get("safety_violation"))
    latest_speed_mps = _latest_speed(rows)

    risk_score = 0
    reasons = []
    allowed = True

    # A safe state should prefer task progress over stopping.
    # Stop is still allowed, but it should not dominate merely because its risk is zero.
    clearance_known_and_safe = (
        min_dist is not None and min_dist > safety_radius_m
    )
    speed_known_or_ok = (
        latest_speed_mps is None or latest_speed_mps <= speed_limit_mps
    )
    state_safe_for_progress = (
        clearance_known_and_safe
        and not had_contact
        and not safety_violation
        and speed_known_or_ok
    )

    if category == "stop":
        if state_safe_for_progress:
            risk_score += 30
            reasons.append(
                "Stop is safe but unnecessarily halts task progress because the current state is outside the safety radius."
            )
        else:
            risk_score += 0
            reasons.append("Stop or hold action is the safest fallback.")
    elif category == "back":
        risk_score += 15
        reasons.append("Backing away may increase clearance, but rear space is not verified by this log.")
    elif category == "turn":
        risk_score += 20
        reasons.append("Turning in place is usually safer than moving forward, but rotation clearance is not fully verified.")
    elif category == "forward":
        risk_score += 40
        reasons.append("Forward or approach action may reduce clearance.")
    elif category == "manipulation":
        risk_score += 35
        reasons.append("Manipulation can disturb nearby objects, so it needs object and contact checks.")
    elif category == "policy":
        if state_safe_for_progress:
            risk_score += 5
            reasons.append(
                "Raw policy action is preferred because the current state is outside the safety radius with no active contact or violation."
            )
        else:
            risk_score += 25
            reasons.append("Raw policy action is allowed only when the current state is not already unsafe.")
    else:
        risk_score += 45
        reasons.append("Action type is unknown, so it is treated conservatively.")

    if had_contact:
        risk_score += 40
        reasons.append("Current or recent step shows contact with object bodies.")

        if category in ("forward", "manipulation", "unknown", "policy"):
            allowed = False
            reasons.append("Action is blocked because contact is already present.")

    if min_dist is not None:
        if min_dist <= hard_stop_radius_m:
            risk_score += 60
            reasons.append(
                f"Closest distance {min_dist:.3f} m is inside hard stop radius {hard_stop_radius_m:.3f} m."
            )

            if category not in ("stop", "back", "turn"):
                allowed = False
                reasons.append("Only stop, back, or turn should be considered inside hard stop radius.")

        elif min_dist <= safety_radius_m:
            risk_score += 35
            reasons.append(
                f"Closest distance {min_dist:.3f} m is inside safety radius {safety_radius_m:.3f} m."
            )

            if category in ("forward", "manipulation", "unknown", "policy"):
                allowed = False
                reasons.append("Action is blocked because it may worsen a safety radius violation.")

        else:
            reasons.append(
                f"Closest distance {min_dist:.3f} m is outside safety radius {safety_radius_m:.3f} m."
            )
    else:
        risk_score += 20
        reasons.append("No closest distance was logged, so clearance is unknown.")

        if category in ("forward", "manipulation", "unknown", "policy"):
            allowed = False
            reasons.append("Action is blocked because clearance is unknown.")

    if safety_violation:
        risk_score += 30
        reasons.append("Latest step is marked as a safety violation.")

        if category in ("forward", "manipulation", "unknown", "policy"):
            allowed = False
            reasons.append("Action is blocked because latest state is already unsafe.")

    if latest_speed_mps is not None:
        if latest_speed_mps > speed_limit_mps:
            risk_score += 25
            reasons.append(
                f"Latest speed {latest_speed_mps:.3f} m/s exceeds speed limit {speed_limit_mps:.3f} m/s."
            )

            if category not in ("stop", "back", "turn"):
                allowed = False
                reasons.append("Action is blocked because robot is already moving too fast.")
        else:
            reasons.append(
                f"Latest speed {latest_speed_mps:.3f} m/s is within speed limit {speed_limit_mps:.3f} m/s."
            )
    else:
        reasons.append("Latest speed could not be computed from logged positions.")

    risk_score = max(0, min(100, risk_score))

    if risk_score >= 75:
        risk_level = "high"
    elif risk_score >= 35:
        risk_level = "medium"
    else:
        risk_level = "low"

    return {
        "action": action,
        "action_text": info["action_text"],
        "category": category,
        "allowed": allowed,
        "risk_level": risk_level,
        "risk_score_0_to_100": risk_score,
        "closest_object": closest_object,
        "min_dist_m": min_dist,
        "had_contact": had_contact,
        "safety_violation": safety_violation,
        "latest_speed_mps": latest_speed_mps,
        "reasons": reasons,
    }


def _safe_action_sort_key(item: Dict[str, Any]) -> Tuple[int, int]:
    allowed_rank = 0 if item.get("allowed") else 1
    risk_score = item.get("risk_score_0_to_100")
    if risk_score is None:
        risk_score = 100
    return allowed_rank, int(risk_score)


# ============================================================
# Existing audit tools
# ============================================================

@tool
def safety_run_summary(log_path: str) -> Dict[str, Any]:
    """
    User tool: summarize whether the robot run looked safe overall.
    Use this first before deeper inspection.
    """
    rows = _load_jsonl(log_path)
    steps = _step_rows(rows)

    if not steps:
        return {"error": "No step rows found."}

    violations = [r for r in steps if r.get("safety_violation")]
    contacts = [r for r in steps if _is_contact_step(r)]

    min_dist = min(
        [r.get("min_dist") for r in steps if r.get("min_dist") is not None],
        default=None,
    )

    return {
        "num_steps": len(steps),
        "duration_wall_sec": steps[-1].get("t_wall"),
        "had_safety_violation": len(violations) > 0,
        "num_safety_violation_steps": len(violations),
        "had_contact": len(contacts) > 0,
        "num_contact_steps": len(contacts),
        "min_distance_m": min_dist,
        "final_collision_count": steps[-1].get("total_collision_events"),
        "unique_collisions_count": steps[-1].get("unique_collisions_count"),
    }


@tool
def extract_collided_objects(log_path: str) -> Dict[str, Any]:
    """
    User tool: return objects the robot collided with or contacted.
    Use when asking: what did the robot hit?
    """
    rows = _load_jsonl(log_path)
    steps = _step_rows(rows)

    collided = {}

    for r in steps:
        step = r.get("step")
        t = r.get("t_wall")

        for c in r.get("contact_bodies_top5", []) or []:
            name = str(c)

            if name not in collided:
                collided[name] = {
                    "first_step": step,
                    "first_t_wall": t,
                    "num_steps_seen": 0,
                }

            collided[name]["num_steps_seen"] += 1

    return {
        "num_collided_objects": len(collided),
        "collided_objects": collided,
    }


@tool
def extract_velocity_profile(log_path: str) -> Dict[str, Any]:
    """
    User tool: compute speed profile from robot positions over time.
    Use when asking: how fast was the robot moving?
    """
    rows = _load_jsonl(log_path)
    speeds = _compute_velocity_samples(rows)
    vals = [s["speed_mps"] for s in speeds]

    return {
        "num_velocity_samples": len(vals),
        "max_speed_mps": max(vals) if vals else None,
        "mean_speed_mps": sum(vals) / len(vals) if vals else None,
        "last_speed_mps": vals[-1] if vals else None,
        "top_speed_samples": sorted(
            speeds,
            key=lambda x: x["speed_mps"],
            reverse=True,
        )[:10],
    }


@tool
def extract_near_misses(log_path: str, danger_radius_m: float = 0.75) -> Dict[str, Any]:
    """
    User tool: find objects that came too close without necessarily colliding.
    Use when asking: what near misses happened?
    """
    rows = _load_jsonl(log_path)
    steps = _step_rows(rows)

    near = {}

    for r in steps:
        for item in _get_nearby_list(r):
            name = item.get("name")
            dist = item.get("dist")

            if name is None or dist is None:
                continue

            if dist <= danger_radius_m:
                if name not in near:
                    near[name] = {
                        "min_dist": dist,
                        "first_step": r.get("step"),
                        "first_t_wall": r.get("t_wall"),
                        "num_near_steps": 0,
                    }

                near[name]["num_near_steps"] += 1
                near[name]["min_dist"] = min(near[name]["min_dist"], dist)

    return {
        "danger_radius_m": danger_radius_m,
        "num_near_miss_objects": len(near),
        "near_miss_objects": dict(
            sorted(near.items(), key=lambda kv: kv[1]["min_dist"])
        ),
    }


@tool
def extract_safety_violations(log_path: str) -> Dict[str, Any]:
    """
    User tool: return when and why safety radius violations occurred.
    Use when asking: when was the robot unsafe?
    """
    rows = _load_jsonl(log_path)
    steps = _step_rows(rows)

    violations = []

    for r in steps:
        if r.get("safety_violation"):
            violations.append({
                "step": r.get("step"),
                "t_wall": r.get("t_wall"),
                "closest_object": r.get("closest_object"),
                "min_dist": r.get("min_dist"),
                "robot_pos": r.get("robot_pos"),
            })

    return {
        "num_violation_steps": len(violations),
        "first_violation": violations[0] if violations else None,
        "worst_violation": min(
            violations,
            key=lambda x: x["min_dist"] if x["min_dist"] is not None else float("inf"),
        ) if violations else None,
        "sample_violations": violations[:20],
    }


@tool
def extract_object_displacements(log_path: str) -> Dict[str, Any]:
    """
    User tool: detect objects that moved during the episode.
    Use when asking: did the robot disturb the scene?
    Requires enhanced logger rows with object_displacements or final_summary.
    """
    rows = _load_jsonl(log_path)

    summaries = [
        r for r in rows
        if r.get("type") in ("summary", "final_summary")
    ]

    for s in reversed(summaries):
        if "object_displacements" in s:
            return {
                "object_displacements": s["object_displacements"],
            }

    steps = _step_rows(rows)
    moved = []

    for r in reversed(steps):
        if "object_displacements" in r:
            moved = r["object_displacements"]
            break

    return {
        "object_displacements": moved,
        "note": "No object displacement summary found unless enhanced logger recorded it.",
    }


@tool
def extract_interventions(log_path: str) -> Dict[str, Any]:
    """
    User tool: return safety interventions, action overrides, emergency stops,
    blocked actions, or fallback policy events.
    Use when asking: did the safety framework intervene?
    """
    rows = _load_jsonl(log_path)

    interventions = [
        r for r in rows
        if r.get("type") in (
            "intervention",
            "safety_intervention",
            "action_override",
            "emergency_stop",
            "fallback_policy",
            "blocked_action",
        )
    ]

    return {
        "num_interventions": len(interventions),
        "interventions": interventions[:50],
    }


@tool
def extract_action_modifications(log_path: str) -> Dict[str, Any]:
    """
    User tool: compare raw policy actions to safety filtered actions.
    Use when asking: did the safety layer change the robot command?
    Requires enhanced logger rows of type action.
    """
    rows = _load_jsonl(log_path)

    action_rows = [r for r in rows if r.get("type") == "action"]
    modified = []

    for r in action_rows:
        raw = r.get("raw_action")
        safe = r.get("safe_action")

        if raw != safe:
            modified.append(r)

    return {
        "num_action_rows": len(action_rows),
        "num_modified_actions": len(modified),
        "modified_action_fraction": (
            len(modified) / len(action_rows) if action_rows else None
        ),
        "sample_modified_actions": modified[:20],
    }


@tool
def extract_task_outcome(log_path: str) -> Dict[str, Any]:
    """
    User tool: determine whether the task succeeded, failed, timed out,
    or terminated for safety reasons.
    """
    rows = _load_jsonl(log_path)

    outcome_rows = [
        r for r in rows
        if r.get("type") in (
            "task_outcome",
            "episode_end",
            "final_summary",
            "summary",
        )
    ]

    if outcome_rows:
        return outcome_rows[-1]

    steps = _step_rows(rows)

    return {
        "task_outcome_found": False,
        "last_step": steps[-1].get("step") if steps else None,
        "duration_wall_sec": steps[-1].get("t_wall") if steps else None,
        "note": "No explicit task outcome row found. Add logger.log_task_outcome(...).",
    }


@tool
def user_safety_brief(log_path: str) -> Dict[str, Any]:
    """
    User tool: compact mission safety brief.
    Use this when the LLM needs one high level answer:
    Was the run safe, what happened, and what should a user care about?
    """
    summary = safety_run_summary.invoke({"log_path": log_path})
    collisions = extract_collided_objects.invoke({"log_path": log_path})
    near_misses = extract_near_misses.invoke({"log_path": log_path})
    velocity = extract_velocity_profile.invoke({"log_path": log_path})
    interventions = extract_interventions.invoke({"log_path": log_path})
    task = extract_task_outcome.invoke({"log_path": log_path})

    concerns = []

    if summary.get("had_contact"):
        concerns.append("Robot made contact with objects.")

    if summary.get("had_safety_violation"):
        concerns.append("Robot entered safety radius.")

    if interventions.get("num_interventions", 0) == 0:
        concerns.append("No safety intervention was logged.")

    if task.get("success") is False:
        concerns.append("Task failed.")

    return {
        "safety_summary": summary,
        "collisions": collisions,
        "near_misses": near_misses,
        "velocity": velocity,
        "interventions": interventions,
        "task_outcome": task,
        "user_concerns": concerns,
    }


# ============================================================
# New commander tools
# These are the tools the LLM should use before choosing action.
# ============================================================

@tool
def get_latest_robot_state(log_path: str) -> Dict[str, Any]:
    """
    Commander tool: return the latest robot state from the log.
    Use this before making any decision.
    The LLM must not invent robot pose, contacts, closest object, or safety state.
    """
    rows = _load_jsonl(log_path)
    latest = _latest_step(rows)

    if latest is None:
        return {
            "error": "No step rows found.",
            "decision_guidance": "State is unknown. Choose safest fallback if control is required.",
        }

    latest_speed = _latest_speed(rows)
    nearby = _get_nearby_list(latest)

    return {
        "step": latest.get("step"),
        "t_wall": latest.get("t_wall"),
        "robot_pos": latest.get("robot_pos"),
        "robot_quat": latest.get("robot_quat"),
        "robot_euler": latest.get("robot_euler"),
        "closest_object": latest.get("closest_object"),
        "min_dist_m": latest.get("min_dist"),
        "num_contacts": latest.get("num_contacts"),
        "contact_bodies_top5": latest.get("contact_bodies_top5", []) or [],
        "safety_violation": bool(latest.get("safety_violation")),
        "total_collision_events": latest.get("total_collision_events"),
        "unique_collisions_count": latest.get("unique_collisions_count"),
        "latest_speed_mps": latest_speed,
        "nearby_top": nearby[:10],
        "raw_step_keys": sorted(list(latest.keys())),
    }


@tool
def get_nearby_objects_now(log_path: str, radius_m: float = 1.0) -> Dict[str, Any]:
    """
    Commander tool: return nearby objects around the robot at the latest step.
    Use this to identify collision risk before choosing an action.
    """
    rows = _load_jsonl(log_path)
    latest = _latest_step(rows)

    if latest is None:
        return {
            "error": "No step rows found.",
            "radius_m": radius_m,
            "nearby_objects": [],
        }

    nearby = _get_nearby_list(latest)
    filtered = []

    for item in nearby:
        dist = item.get("dist")

        if dist is None:
            continue

        if dist <= radius_m:
            filtered.append(item)

    return {
        "step": latest.get("step"),
        "t_wall": latest.get("t_wall"),
        "radius_m": radius_m,
        "num_nearby_objects": len(filtered),
        "nearby_objects": filtered,
        "closest_object": latest.get("closest_object"),
        "min_dist_m": latest.get("min_dist"),
    }


@tool
def get_available_actions(log_path: str) -> Dict[str, Any]:
    """
    Commander tool: return actions available to the policy or agent if the log contains them.
    Use this before evaluating candidate actions.
    If no available actions are logged, this tool returns conservative fallback actions.
    """
    rows = _load_jsonl(log_path)
    found = _extract_available_actions_from_rows(rows)

    if found["found"]:
        return {
            "available_actions_found": True,
            "source_type": found["source_type"],
            "source_step": found["source_step"],
            "source_key": found["source_key"],
            "actions": found["actions"],
            "note": "Use evaluate_candidate_actions before choosing among these actions.",
        }

    fallback_actions = [
        "stop",
        "back_up",
        "turn_left",
        "turn_right",
    ]

    return {
        "available_actions_found": False,
        "actions": fallback_actions,
        "note": (
            "No available action list was found in the log. "
            "Returned conservative fallback actions only. "
            "Add available_actions or candidate_actions to action rows for better decisions."
        ),
    }


@tool
def get_task_progress_now(log_path: str) -> Dict[str, Any]:
    """
    Commander tool: inspect current task progress from the latest task, summary, or step rows.
    Use this before deciding whether to continue, stop, or recover.
    """
    rows = _load_jsonl(log_path)

    task_rows = [
        r for r in rows
        if r.get("type") in (
            "task_progress",
            "task_state",
            "task_outcome",
            "episode_end",
            "summary",
            "final_summary",
        )
    ]

    latest = _latest_step(rows)

    if task_rows:
        last_task = task_rows[-1]
        return {
            "task_info_found": True,
            "source_type": last_task.get("type"),
            "task_info": last_task,
            "latest_step": latest.get("step") if latest else None,
        }

    if latest is None:
        return {
            "task_info_found": False,
            "error": "No step rows found.",
        }

    inferred = {}

    for key in [
        "task",
        "task_name",
        "task_phase",
        "phase",
        "success",
        "done",
        "terminated",
        "truncated",
        "reward",
        "progress",
        "subgoal",
        "goal",
    ]:
        if key in latest:
            inferred[key] = latest.get(key)

    return {
        "task_info_found": bool(inferred),
        "source_type": "step",
        "latest_step": latest.get("step"),
        "task_info": inferred,
        "note": "No explicit task progress row found. Add logger.log_task_progress(...).",
    }


@tool
def get_recent_events(log_path: str, last_n_steps: int = 20) -> Dict[str, Any]:
    """
    Commander tool: summarize recent contacts, violations, interventions, and action changes.
    Use this to know whether the robot is recovering from a recent unsafe event.
    """
    rows = _load_jsonl(log_path)
    recent = _recent_steps(rows, last_n_steps=last_n_steps)

    if not recent:
        return {
            "error": "No recent step rows found.",
            "last_n_steps": last_n_steps,
        }

    first_step = recent[0].get("step")
    last_step = recent[-1].get("step")

    contacts = []
    violations = []

    for r in recent:
        if _is_contact_step(r):
            contacts.append({
                "step": r.get("step"),
                "t_wall": r.get("t_wall"),
                "num_contacts": r.get("num_contacts"),
                "contact_bodies_top5": r.get("contact_bodies_top5", []) or [],
            })

        if r.get("safety_violation"):
            violations.append({
                "step": r.get("step"),
                "t_wall": r.get("t_wall"),
                "closest_object": r.get("closest_object"),
                "min_dist": r.get("min_dist"),
                "robot_pos": r.get("robot_pos"),
            })

    action_rows = [
        r for r in rows
        if r.get("type") == "action"
    ]

    recent_action_rows = []
    for r in action_rows:
        s = r.get("step")
        if s is None:
            continue
        if first_step is not None and last_step is not None and first_step <= s <= last_step:
            recent_action_rows.append(r)

    modified_actions = []
    for r in recent_action_rows:
        if r.get("raw_action") != r.get("safe_action"):
            modified_actions.append(r)

    intervention_rows = [
        r for r in rows
        if r.get("type") in (
            "intervention",
            "safety_intervention",
            "action_override",
            "emergency_stop",
            "fallback_policy",
            "blocked_action",
        )
    ]

    recent_interventions = []
    for r in intervention_rows:
        s = r.get("step")
        if s is None:
            recent_interventions.append(r)
        elif first_step is not None and last_step is not None and first_step <= s <= last_step:
            recent_interventions.append(r)

    return {
        "last_n_steps": last_n_steps,
        "first_step": first_step,
        "last_step": last_step,
        "num_recent_contact_steps": len(contacts),
        "recent_contacts": contacts[:20],
        "num_recent_violation_steps": len(violations),
        "recent_violations": violations[:20],
        "num_recent_action_rows": len(recent_action_rows),
        "num_recent_modified_actions": len(modified_actions),
        "recent_modified_actions": modified_actions[:20],
        "num_recent_interventions": len(recent_interventions),
        "recent_interventions": recent_interventions[:20],
    }


@tool
def evaluate_candidate_actions(
    log_path: str,
    candidate_actions: List[Any],
    safety_radius_m: float = 0.75,
    hard_stop_radius_m: float = 0.25,
    speed_limit_mps: float = 1.0,
) -> Dict[str, Any]:
    """
    Commander tool: evaluate candidate actions using latest logged state.
    The LLM should call this before choosing an action.
    The LLM should only choose actions marked allowed=true.
    """
    rows = _load_jsonl(log_path)
    latest = _latest_step(rows)

    if latest is None:
        return {
            "error": "No step rows found.",
            "evaluations": [],
            "decision_rule": "State unknown. Choose stop if available.",
        }

    if not candidate_actions:
        found = _extract_available_actions_from_rows(rows)

        if found["found"]:
            candidate_actions = found["actions"]
        else:
            candidate_actions = ["stop", "back_up", "turn_left", "turn_right"]

    evaluations = []

    for action in candidate_actions:
        ev = _risk_from_state_and_action(
            latest=latest,
            rows=rows,
            action=action,
            safety_radius_m=safety_radius_m,
            hard_stop_radius_m=hard_stop_radius_m,
            speed_limit_mps=speed_limit_mps,
        )
        evaluations.append(ev)

    evaluations_sorted = sorted(evaluations, key=_safe_action_sort_key)

    allowed = [x for x in evaluations_sorted if x.get("allowed")]
    blocked = [x for x in evaluations_sorted if not x.get("allowed")]

    return {
        "step": latest.get("step"),
        "t_wall": latest.get("t_wall"),
        "safety_radius_m": safety_radius_m,
        "hard_stop_radius_m": hard_stop_radius_m,
        "speed_limit_mps": speed_limit_mps,
        "closest_object": latest.get("closest_object"),
        "min_dist_m": latest.get("min_dist"),
        "safety_violation": bool(latest.get("safety_violation")),
        "num_contacts": latest.get("num_contacts"),
        "evaluations": evaluations_sorted,
        "allowed_actions": allowed,
        "blocked_actions": blocked,
        "decision_rule": (
            "If the current state is safe and the raw policy action is allowed, prefer task progress. "
            "Choose stop or recovery actions when the state is unsafe, clearance is too small, contact is present, "
            "or no progress action is allowed."
        ),
    }


@tool
def choose_safe_action(
    log_path: str,
    candidate_actions: List[Any],
    safety_radius_m: float = 0.75,
    hard_stop_radius_m: float = 0.25,
    speed_limit_mps: float = 1.0,
) -> Dict[str, Any]:
    """
    Commander tool: choose the safest action from candidate actions.
    Use this when the LLM should not make the final safety choice itself.
    """
    result = evaluate_candidate_actions.invoke({
        "log_path": log_path,
        "candidate_actions": candidate_actions,
        "safety_radius_m": safety_radius_m,
        "hard_stop_radius_m": hard_stop_radius_m,
        "speed_limit_mps": speed_limit_mps,
    })

    if result.get("error"):
        return {
            "chosen_action": "stop",
            "chosen_reason": "Could not evaluate state. Stop is safest fallback.",
            "evaluation": result,
        }

    allowed = result.get("allowed_actions", [])

    if allowed:
        chosen = allowed[0]

        return {
            "chosen_action": chosen.get("action"),
            "chosen_action_text": chosen.get("action_text"),
            "chosen_category": chosen.get("category"),
            "chosen_risk_level": chosen.get("risk_level"),
            "chosen_risk_score_0_to_100": chosen.get("risk_score_0_to_100"),
            "chosen_reason": chosen.get("reasons"),
            "evaluation": result,
        }

    fallback = None

    for ev in result.get("evaluations", []):
        if ev.get("category") == "stop":
            fallback = ev
            break

    if fallback is not None:
        return {
            "chosen_action": fallback.get("action"),
            "chosen_action_text": fallback.get("action_text"),
            "chosen_category": fallback.get("category"),
            "chosen_risk_level": fallback.get("risk_level"),
            "chosen_risk_score_0_to_100": fallback.get("risk_score_0_to_100"),
            "chosen_reason": (
                fallback.get("reasons", []) +
                ["No allowed action was found, but stop is the safest fallback."]
            ),
            "evaluation": result,
        }

    return {
        "chosen_action": "stop",
        "chosen_action_text": "stop",
        "chosen_category": "stop",
        "chosen_risk_level": "low",
        "chosen_risk_score_0_to_100": 0,
        "chosen_reason": "No allowed action was found and no stop action was provided. Returning stop.",
        "evaluation": result,
    }


@tool
def get_raw_log_window(
    log_path: str,
    center_step: Optional[int] = None,
    window: int = 5,
    row_types: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Commander/debug tool: return raw log rows around a step.
    Use this only when the LLM needs to verify a surprising event.
    Do not use this as the first tool unless debugging.
    """
    rows = _load_jsonl(log_path)
    steps = _step_rows(rows)

    if not rows:
        return {
            "error": "Log file is empty.",
            "rows": [],
        }

    if center_step is None:
        latest = _latest_step(rows)
        center_step = latest.get("step") if latest else None

    if center_step is None:
        selected = rows[-max(1, window):]
        return {
            "center_step": None,
            "window": window,
            "row_types": row_types,
            "num_rows": len(selected),
            "rows": selected,
            "note": "No step number found. Returned last rows instead.",
        }

    selected = []

    for r in rows:
        if row_types is not None and r.get("type") not in row_types:
            continue

        s = r.get("step")

        if s is None:
            continue

        if center_step - window <= s <= center_step + window:
            selected.append(r)

    return {
        "center_step": center_step,
        "window": window,
        "row_types": row_types,
        "num_rows": len(selected),
        "rows": selected[:200],
        "truncated": len(selected) > 200,
    }


@tool
def inspect_object_history(
    log_path: str,
    object_name: str,
    max_rows: int = 50,
) -> Dict[str, Any]:
    """
    Commander/debug tool: inspect how a specific object appeared in nearby object logs,
    closest object logs, and contact logs over time.
    Use when asking what happened with one object.
    """
    rows = _load_jsonl(log_path)
    steps = _step_rows(rows)

    matches = []

    obj_lower = object_name.lower()

    for r in steps:
        found = False
        reason = []

        closest = r.get("closest_object")
        if closest is not None and obj_lower in str(closest).lower():
            found = True
            reason.append("closest_object")

        contacts = r.get("contact_bodies_top5", []) or []
        matched_contacts = [
            c for c in contacts
            if obj_lower in str(c).lower()
        ]

        if matched_contacts:
            found = True
            reason.append("contact_bodies_top5")

        nearby_matches = []

        for item in _get_nearby_list(r):
            name = item.get("name")
            if name is not None and obj_lower in str(name).lower():
                found = True
                reason.append("nearby_top")
                nearby_matches.append(item)

        if found:
            matches.append({
                "step": r.get("step"),
                "t_wall": r.get("t_wall"),
                "robot_pos": r.get("robot_pos"),
                "closest_object": closest,
                "min_dist": r.get("min_dist"),
                "safety_violation": r.get("safety_violation"),
                "matched_contacts": matched_contacts,
                "nearby_matches": nearby_matches,
                "reason": sorted(list(set(reason))),
            })

    return {
        "object_name_query": object_name,
        "num_matching_steps": len(matches),
        "first_match": matches[0] if matches else None,
        "last_match": matches[-1] if matches else None,
        "sample_matches": matches[:max_rows],
        "truncated": len(matches) > max_rows,
    }


@tool
def commander_state_brief(log_path: str) -> Dict[str, Any]:
    """
    Commander tool: compact current state brief.
    Use this when the LLM needs one grounded state summary before action selection.
    """
    latest = get_latest_robot_state.invoke({"log_path": log_path})
    nearby = get_nearby_objects_now.invoke({"log_path": log_path, "radius_m": 1.0})
    actions = get_available_actions.invoke({"log_path": log_path})
    task = get_task_progress_now.invoke({"log_path": log_path})
    recent = get_recent_events.invoke({"log_path": log_path, "last_n_steps": 20})

    warnings = []

    if latest.get("safety_violation"):
        warnings.append("Latest state is marked as a safety violation.")

    if latest.get("num_contacts") and latest.get("num_contacts") > 0:
        warnings.append("Latest state has active contacts.")

    if nearby.get("num_nearby_objects", 0) > 0:
        warnings.append("Objects are within 1.0 m of the robot.")

    if recent.get("num_recent_contact_steps", 0) > 0:
        warnings.append("There were recent contact steps.")

    if recent.get("num_recent_violation_steps", 0) > 0:
        warnings.append("There were recent safety violation steps.")

    if not actions.get("available_actions_found"):
        warnings.append("No real available action list was logged. Using fallback actions only.")

    return {
        "latest_robot_state": latest,
        "nearby_objects_now": nearby,
        "available_actions": actions,
        "task_progress_now": task,
        "recent_events": recent,
        "commander_warnings": warnings,
        "decision_protocol": [
            "Do not invent world state.",
            "Evaluate candidate actions using evaluate_candidate_actions.",
            "Choose only an allowed action.",
            "If unsure, choose stop.",
        ],
    }


# ============================================================
# Tool groups
# ============================================================

AUDIT_LOG_TOOLS = [
    safety_run_summary,
    extract_collided_objects,
    extract_velocity_profile,
    extract_near_misses,
    extract_safety_violations,
    extract_object_displacements,
    extract_interventions,
    extract_action_modifications,
    extract_task_outcome,
    user_safety_brief,
]

COMMANDER_LOG_TOOLS = [
    get_latest_robot_state,
    get_nearby_objects_now,
    get_available_actions,
    get_task_progress_now,
    get_recent_events,
    evaluate_candidate_actions,
    choose_safe_action,
    get_raw_log_window,
    inspect_object_history,
    commander_state_brief,
]

USER_LOG_TOOLS = AUDIT_LOG_TOOLS + COMMANDER_LOG_TOOLS

@tool
def evaluate_generated_vcio_scores(
    log_path: str,
    scorers_dir: str = "TRACE/generated_vcio",
) -> str:
    """
    Run generated VCIO Python scorers on a TRACE SafetyLogger JSONL file.

    Returns aggregate VCIO safety scores and evidence from each generated scorer.
    Lower score means safer; higher score means more severe violation.
    """
    import json

    from TRACE.eval.generated_vcio_runner import run_generated_vcio_scorers

    result = run_generated_vcio_scorers(
        log_path=log_path,
        scorers_dir=scorers_dir,
    )

    return json.dumps(result, indent=2)


@tool
def evaluate_generated_vcio_scores(
    log_path: str,
    scorers_dir: str = "TRACE/generated_vcio",
) -> str:
    """
    Run generated VCIO Python scorers on a TRACE SafetyLogger JSONL file.

    Returns aggregate VCIO safety scores and evidence from each generated scorer.
    Lower score means safer; higher score means more severe violation.
    """
    import json

    from TRACE.eval.generated_vcio_runner import run_generated_vcio_scorers

    result = run_generated_vcio_scorers(
        log_path=log_path,
        scorers_dir=scorers_dir,
    )

    return json.dumps(result, indent=2)
