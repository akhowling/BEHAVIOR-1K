import argparse
import json
import sys
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path("~/llm/.env").expanduser())

from TRACE.logging.safety_log_tools import COMMANDER_LOG_TOOLS


# ============================================================
# Tool registry
# ============================================================

TOOL_REGISTRY = {t.name: t for t in COMMANDER_LOG_TOOLS}


def call_tool(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calls one of the LangChain @tool tools safely.
    The LLM never directly executes code. It can only receive outputs from here.
    """
    if tool_name not in TOOL_REGISTRY:
        return {
            "error": f"Unknown tool: {tool_name}",
            "available_tools": sorted(TOOL_REGISTRY.keys()),
        }

    try:
        result = TOOL_REGISTRY[tool_name].invoke(args)
        return {
            "tool": tool_name,
            "args": args,
            "ok": True,
            "result": result,
        }
    except Exception as e:
        return {
            "tool": tool_name,
            "args": args,
            "ok": False,
            "error": repr(e),
        }


# ============================================================
# Ollama client
# ============================================================

def ollama_chat_json(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """
    Calls PARCC/LiteLLM when LLM_PROVIDER=parcc or PARCC_URL is set.
    Otherwise falls back to Ollama.
    """

    provider = os.getenv("LLM_PROVIDER", "").lower().strip()

    parcc_url = os.getenv("PARCC_URL")
    parcc_key = os.getenv("PARCC_API_KEY")
    parcc_model = (
        os.getenv("PARCC_MODEL")
        or os.getenv("OPENAI_MODEL")
        or os.getenv("OLLAMA_MODEL")
        or "openai/gpt-oss-20b"
    )

    if provider == "parcc" or parcc_url:
        if not parcc_url:
            return {"ok": False, "error": "PARCC_URL is not set."}

        if not parcc_key:
            return {"ok": False, "error": "PARCC_API_KEY is not set."}

        base_url = parcc_url.rstrip("/")
        if base_url.endswith("/v1"):
            url = f"{base_url}/chat/completions"
        else:
            url = f"{base_url}/v1/chat/completions"

        payload = {
            "model": parcc_model,
            "temperature": temperature,
            "max_tokens": 512,
            "reasoning_effort": "low",
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt + "\nReturn only valid JSON. Do not include markdown.",
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
        }

        data = json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {parcc_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            try:
                body = e.read().decode("utf-8")
            except Exception:
                body = ""
            return {
                "ok": False,
                "error": f"PARCC HTTP error: {e}",
                "body": body,
            }
        except Exception as e:
            return {
                "ok": False,
                "error": f"Could not reach PARCC: {repr(e)}",
            }

        try:
            outer = json.loads(raw)
        except json.JSONDecodeError:
            return {
                "ok": False,
                "error": "PARCC returned non JSON response.",
                "raw": raw,
            }

        try:
            content = outer["choices"][0]["message"]["content"]
        except Exception:
            return {
                "ok": False,
                "error": "PARCC response did not contain choices[0].message.content.",
                "raw_response": outer,
            }

        if isinstance(content, dict):
            parsed = content
        else:
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                left = content.find("{")
                right = content.rfind("}")
                if left >= 0 and right > left:
                    try:
                        parsed = json.loads(content[left:right + 1])
                    except json.JSONDecodeError:
                        return {
                            "ok": False,
                            "error": "PARCC model did not return valid JSON content.",
                            "raw_content": content,
                            "raw_response": outer,
                        }
                else:
                    return {
                        "ok": False,
                        "error": "PARCC model did not return valid JSON content.",
                        "raw_content": content,
                        "raw_response": outer,
                    }

        return {
            "ok": True,
            "json": parsed,
            "raw_response": outer,
        }

    # Default fallback: Ollama.
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    model = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")

    user = os.getenv("OLLAMA_USER")
    password = os.getenv("OLLAMA_PASSWORD")

    url = f"{base_url}/api/chat"

    payload = {
        "model": model,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": temperature,
            "num_ctx": 8192,
        },
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
    }

    if user and password:
        payload["user"] = user
        payload["password"] = password

    data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.URLError as e:
        return {
            "ok": False,
            "error": f"Could not reach Ollama: {e}",
        }

    try:
        outer = json.loads(raw)
    except json.JSONDecodeError:
        return {
            "ok": False,
            "error": "Ollama returned non JSON response.",
            "raw": raw,
        }

    content = outer.get("message", {}).get("content", "")

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return {
            "ok": False,
            "error": "Model did not return valid JSON content.",
            "raw_content": content,
            "raw_response": outer,
        }

    return {
        "ok": True,
        "json": parsed,
        "raw_response": outer,
    }


def json_dumps(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, default=str)


def compact_state_brief(state_brief: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep the LLM input small and grounded.
    Do not feed the full log.(or do?)
    """

    result = state_brief.get("result", {})
    latest = result.get("latest_robot_state", {})
    nearby = result.get("nearby_objects_now", {})
    task = result.get("task_progress_now", {})
    recent = result.get("recent_events", {})
    warnings = result.get("commander_warnings", [])

    return {
        "latest_robot_state": {
            "step": latest.get("step"),
            "t_wall": latest.get("t_wall"),
            "robot_pos": latest.get("robot_pos"),
            "closest_object": latest.get("closest_object"),
            "min_dist_m": latest.get("min_dist_m"),
            "num_contacts": latest.get("num_contacts"),
            "contact_bodies_top5": latest.get("contact_bodies_top5"),
            "safety_violation": latest.get("safety_violation"),
            "latest_speed_mps": latest.get("latest_speed_mps"),
        },
        "nearby_objects_now": {
            "radius_m": nearby.get("radius_m"),
            "num_nearby_objects": nearby.get("num_nearby_objects"),
            "nearby_objects": nearby.get("nearby_objects", [])[:10],
        },
        "task_progress_now": task,
        "recent_events": {
            "last_n_steps": recent.get("last_n_steps"),
            "num_recent_contact_steps": recent.get("num_recent_contact_steps"),
            "num_recent_violation_steps": recent.get("num_recent_violation_steps"),
            "num_recent_modified_actions": recent.get("num_recent_modified_actions"),
            "num_recent_interventions": recent.get("num_recent_interventions"),
            "recent_contacts": recent.get("recent_contacts", [])[:5],
            "recent_violations": recent.get("recent_violations", [])[:5],
            "recent_interventions": recent.get("recent_interventions", [])[:5],
        },
        "commander_warnings": warnings,
    }


def build_action_catalog(evaluation_tool_output: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Converts tool action evaluations into stable IDs so the LLM chooses a0, a1, etc.
    This avoids ambiguity when actions are dicts, arrays, or long strings.
    """

    result = evaluation_tool_output.get("result", {})
    evaluations = result.get("evaluations", [])

    catalog = []

    for i, ev in enumerate(evaluations):
        catalog.append({
            "id": f"a{i}",
            "action": ev.get("action"),
            "action_text": ev.get("action_text"),
            "category": ev.get("category"),
            "allowed": ev.get("allowed"),
            "risk_level": ev.get("risk_level"),
            "risk_score_0_to_100": ev.get("risk_score_0_to_100"),
            "closest_object": ev.get("closest_object"),
            "min_dist_m": ev.get("min_dist_m"),
            "had_contact": ev.get("had_contact"),
            "safety_violation": ev.get("safety_violation"),
            "latest_speed_mps": ev.get("latest_speed_mps"),
            "reasons": ev.get("reasons", [])[:6],
        })

    return catalog


def get_allowed_action_ids(action_catalog: List[Dict[str, Any]]) -> List[str]:
    return [a["id"] for a in action_catalog if a.get("allowed") is True]


def choose_tool_fallback(action_catalog: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Deterministic fallback if the LLM gives an invalid answer.
    Prefer lowest risk allowed action.
    If none allowed, prefer stop.
    """

    allowed = [a for a in action_catalog if a.get("allowed") is True]

    if allowed:
        allowed.sort(key=lambda x: x.get("risk_score_0_to_100", 100))
        return allowed[0]

    for a in action_catalog:
        if a.get("category") == "stop":
            return a

    return None


# ============================================================
# System prompt
# ============================================================

SYSTEM_PROMPT = """
You are a robot safety commander.

You must follow these rules:

1. You are not the source of truth.
2. The tool outputs are the source of truth.
3. You must not invent object positions, collisions, task success, safety violations, or available actions.
4. You must choose only from the provided action_catalog.
5. You must choose only an action with allowed=true.
6. If the current state is safe and a raw policy/progress action is allowed, choose the progress action instead of stop.
7. If the current state is unsafe, contact is present, clearance is too small, or evidence is unclear, choose the safest stop or hold action if it exists.
8. Do not provide hidden chain of thought.
8. Give a short evidence based rationale only.
9. Your output must be valid JSON.

Your JSON format must be:

{
  "chosen_action_id": "a0",
  "confidence": "low | medium | high",
  "safety_verdict": "safe | caution | unsafe",
  "rationale": "Short explanation based only on tool evidence.",
  "evidence_used": [
    "Specific evidence from tool outputs."
  ],
  "uncertainties": [
    "Anything important that the tools did not prove."
  ]
}
"""


# ============================================================
# Commander loop
# ============================================================

def run_commander_once(
    log_path: str,
    candidate_actions: Optional[List[Any]] = None,
    mission: Optional[str] = None,
    include_raw_window: bool = True,
    safety_radius_m: float = 0.75,
    hard_stop_radius_m: float = 0.25,
    speed_limit_mps: float = 1.0,
) -> Dict[str, Any]:
    """
    One complete commander decision cycle.

    Protocol:
    1. Inspect current state through tools.
    2. Get available actions through tools if user did not provide candidates.
    3. Evaluate candidates through tools.
    4. Optionally inspect a small raw log window.
    5. Ask Ollama to choose from allowed actions only.
    6. Validate model output.
    7. If invalid, use deterministic tool fallback.
    """

    tool_trace = []

    # 1. Current state brief
    state_brief = call_tool(
        "commander_state_brief",
        {
            "log_path": log_path,
        },
    )
    tool_trace.append(state_brief)

    compact_state = compact_state_brief(state_brief)

    # 2. Candidate actions
    if candidate_actions is None:
        available = call_tool(
            "get_available_actions",
            {
                "log_path": log_path,
            },
        )
        tool_trace.append(available)

        candidate_actions = available.get("result", {}).get("actions", [])

    if not candidate_actions:
        candidate_actions = ["stop", "back_up", "turn_left", "turn_right"]

    # 3. Evaluate actions
    evaluation = call_tool(
        "evaluate_candidate_actions",
        {
            "log_path": log_path,
            "candidate_actions": candidate_actions,
            "safety_radius_m": safety_radius_m,
            "hard_stop_radius_m": hard_stop_radius_m,
            "speed_limit_mps": speed_limit_mps,
        },
    )
    tool_trace.append(evaluation)

    action_catalog = build_action_catalog(evaluation)
    allowed_ids = get_allowed_action_ids(action_catalog)

    # 4. Optional bounded raw window
    raw_window = None

    if include_raw_window:
        latest_step = (
            compact_state
            .get("latest_robot_state", {})
            .get("step")
        )

        raw_window = call_tool(
            "get_raw_log_window",
            {
                "log_path": log_path,
                "center_step": latest_step,
                "window": 2,
                "row_types": ["step", "action", "intervention", "safety_intervention", "blocked_action"],
            },
        )
        tool_trace.append(raw_window)

    # 5. Build LLM prompt
    evidence_packet = {
        "mission": mission,
        "state_brief": compact_state,
        "action_catalog": action_catalog,
        "allowed_action_ids": allowed_ids,
        "raw_window": raw_window.get("result") if raw_window else None,
        "decision_instruction": (
            "Choose exactly one chosen_action_id from allowed_action_ids. "
            "If the state is safe and an allowed policy/progress action exists, choose that action to continue task progress. "
            "Choose stop only when the state is unsafe, contact is present, clearance is too small, evidence is unclear, "
            "or no progress action is allowed."
        ),
    }

    user_prompt = f"""
Here is the grounded tool evidence.

{json_dumps(evidence_packet)}

Return only valid JSON in the required schema.
"""

    model_result = ollama_chat_json(
    system_prompt=SYSTEM_PROMPT,
    user_prompt=user_prompt,
    temperature=0.0,
    )

    # 6. Validate model output
    fallback = choose_tool_fallback(action_catalog)

    final = {
        "log_path": log_path,
        "model": os.getenv("PARCC_MODEL") or os.getenv("OLLAMA_MODEL", "gpt-oss:20b"),
        "mission": mission,
        "candidate_actions": candidate_actions,
        "action_catalog": action_catalog,
        "allowed_action_ids": allowed_ids,
        "llm_result": model_result,
        "tool_trace": tool_trace,
        "validated": False,
        "chosen_action": None,
        "chosen_action_id": None,
        "used_fallback": False,
        "final_reason": None,
    }

    if not model_result.get("ok"):
        final["used_fallback"] = True
        final["final_reason"] = "LLM failed or returned invalid JSON. Used deterministic tool fallback."

        if fallback:
            final["chosen_action"] = fallback.get("action")
            final["chosen_action_id"] = fallback.get("id")
        else:
            final["chosen_action"] = "stop"
            final["chosen_action_id"] = None

        return final

    llm_json = model_result.get("json", {})
    chosen_id = llm_json.get("chosen_action_id")

    id_to_action = {a["id"]: a for a in action_catalog}

    if chosen_id in id_to_action and chosen_id in allowed_ids:
        chosen = id_to_action[chosen_id]
        final["validated"] = True
        final["chosen_action"] = chosen.get("action")
        final["chosen_action_id"] = chosen_id
        final["final_reason"] = "LLM chose a valid allowed action from tool evaluated options."
        return final

    # 7. Fallback if model hallucinated, chose blocked action, or picked invalid ID
    final["used_fallback"] = True
    final["final_reason"] = (
        f"LLM chose invalid or blocked action_id={chosen_id}. "
        "Used deterministic tool fallback."
    )

    if fallback:
        final["chosen_action"] = fallback.get("action")
        final["chosen_action_id"] = fallback.get("id")
    else:
        final["chosen_action"] = "stop"
        final["chosen_action_id"] = None

    return final


# ============================================================
# CLI
# ============================================================

def parse_candidate_actions(raw: Optional[str]) -> Optional[List[Any]]:
    if raw is None:
        return None

    raw = raw.strip()

    if not raw:
        return None

    try:
        parsed = json.loads(raw)

        if isinstance(parsed, list):
            return parsed

        return [parsed]

    except json.JSONDecodeError:
        return [x.strip() for x in raw.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--log",
        required=True,
        help="Path to safety JSONL log.",
    )


    parser.add_argument(
        "--mission",
        default=None,
        help="Optional task or mission text.",
    )

    parser.add_argument(
        "--candidate-actions",
        default=None,
        help=(
            "Optional candidate actions. "
            "Can be JSON list or comma separated string. "
            "Example: '[\"stop\", \"move_forward\", \"turn_left\"]'"
        ),
    )

    parser.add_argument(
        "--no-raw-window",
        action="store_true",
        help="Disable passing a small raw log window to the LLM.",
    )

    parser.add_argument(
        "--safety-radius-m",
        type=float,
        default=0.75,
    )

    parser.add_argument(
        "--hard-stop-radius-m",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--speed-limit-mps",
        type=float,
        default=1.0,
    )

    args = parser.parse_args()

    candidate_actions = parse_candidate_actions(args.candidate_actions)

    result = run_commander_once(
        log_path=args.log,
        candidate_actions=candidate_actions,
        mission=args.mission,
        include_raw_window=not args.no_raw_window,
        safety_radius_m=args.safety_radius_m,
        hard_stop_radius_m=args.hard_stop_radius_m,
        speed_limit_mps=args.speed_limit_mps,
    )

    print(json_dumps(result))


if __name__ == "__main__":
    main()