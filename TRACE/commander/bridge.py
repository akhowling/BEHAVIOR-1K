import json
from typing import Optional

from TRACE.commander.llm_reasoning import run_commander_once


def make_jsonable(x):
    if x is None or isinstance(x, (bool, int, float, str)):
        return x

    if isinstance(x, dict):
        return {str(k): make_jsonable(v) for k, v in x.items()}

    if isinstance(x, (list, tuple)):
        return [make_jsonable(v) for v in x]

    try:
        import numpy as np
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.integer, np.floating)):
            return x.item()
    except Exception:
        pass

    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy().tolist()
    except Exception:
        pass

    return repr(x)


def zero_like_action(action):
    """
    Conservative stop/hold action with same broad structure as the raw action.
    """
    if action is None:
        return "stop"

    if isinstance(action, (int, float)):
        return 0.0

    if isinstance(action, list):
        return [zero_like_action(x) for x in action]

    if isinstance(action, tuple):
        return tuple(zero_like_action(x) for x in action)

    if isinstance(action, dict):
        return {k: zero_like_action(v) for k, v in action.items()}

    try:
        import numpy as np
        if isinstance(action, np.ndarray):
            return np.zeros_like(action)
    except Exception:
        pass

    try:
        import torch
        if isinstance(action, torch.Tensor):
            return torch.zeros_like(action)
    except Exception:
        pass

    return "stop"


def actions_equal(a, b):
    return json.dumps(make_jsonable(a), sort_keys=True) == json.dumps(make_jsonable(b), sort_keys=True)


def build_candidate_actions(raw_action):
    """
    Candidate set for commander.
    The first option means: execute the policy's original action.
    Other options are safety recovery choices.
    """
    return [
        {
            "name": "execute_policy_action",
            "kind": "raw_policy_action",
            "action": make_jsonable(raw_action),
        },
        "stop",
        "back_up",
        "turn_left",
        "turn_right",
    ]


def materialize_commander_choice(chosen_action, raw_action):
    """
    Converts commander choice into an actual low-level env action.

    For now:
    - execute_policy_action -> raw policy action
    - stop/back_up/turn_left/turn_right -> zero action

    This is conservative and enough for logging/intervention results.
    Later you can map back_up/turn_left/turn_right to real robot controls.
    """
    if isinstance(chosen_action, dict):
        name = str(chosen_action.get("name", "")).lower()
        kind = str(chosen_action.get("kind", "")).lower()

        if name == "execute_policy_action" or kind == "raw_policy_action":
            return raw_action

    if isinstance(chosen_action, str):
        text = chosen_action.lower().strip()

        if text in {
            "execute_policy_action",
            "continue",
            "continue_policy",
            "raw_policy_action",
            "policy",
        }:
            return raw_action

        if text in {
            "stop",
            "halt",
            "hold",
            "wait",
            "emergency_stop",
            "back_up",
            "turn_left",
            "turn_right",
        }:
            return zero_like_action(raw_action)

    return zero_like_action(raw_action)


def run_action_commander(
    log_path: str,
    step: int,
    raw_action,
    mission: Optional[str] = None,
    every_n_steps: int = 20,
    force: bool = False,
    safety_radius_m: float = 0.75,
    hard_stop_radius_m: float = 0.25,
    speed_limit_mps: float = 1.0,
):
    """
    Returns:
        safe_action, commander_result

    If commander is skipped this step:
        returns raw_action, None
    """
    if not force and every_n_steps > 1 and step % every_n_steps != 0:
        return raw_action, None

    candidate_actions = build_candidate_actions(raw_action)

    result = run_commander_once(
        log_path=log_path,
        candidate_actions=candidate_actions,
        mission=mission,
        include_raw_window=False,
        safety_radius_m=safety_radius_m,
        hard_stop_radius_m=hard_stop_radius_m,
        speed_limit_mps=speed_limit_mps,
    )

    chosen_action = result.get("chosen_action")
    safe_action = materialize_commander_choice(chosen_action, raw_action)

    return safe_action, result


def log_commander_result(logger, step: int, raw_action, safe_action, commander_result):
    """
    Writes commander result into SafetyLogger.
    """
    if commander_result is None:
        logger.log_action(
            step=step,
            raw_action=raw_action,
            safe_action=safe_action,
            extra={
                "commander_called": False,
            },
        )
        return

    modified = not actions_equal(raw_action, safe_action)

    logger.log_action(
        step=step,
        raw_action=raw_action,
        safe_action=safe_action,
        extra={
            "commander_called": True,
            "chosen_action": commander_result.get("chosen_action"),
            "chosen_action_id": commander_result.get("chosen_action_id"),
            "validated": commander_result.get("validated"),
            "used_fallback": commander_result.get("used_fallback"),
            "final_reason": commander_result.get("final_reason"),
        },
    )

    if modified:
        logger.log_intervention(
            step=step,
            reason=commander_result.get("final_reason", "Commander modified policy action."),
            raw_action=raw_action,
            safe_action=safe_action,
            intervention_type="action_commander",
            extra={
                "chosen_action": commander_result.get("chosen_action"),
                "chosen_action_id": commander_result.get("chosen_action_id"),
                "validated": commander_result.get("validated"),
                "used_fallback": commander_result.get("used_fallback"),
                "llm_result": commander_result.get("llm_result"),
            },
        )