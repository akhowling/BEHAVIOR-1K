# ~/BEHAVIOR-1K/safety_logger.py
import json
import math
import time
from pathlib import Path


class SafetyLogger:
    """
    JSONL safety logger for OmniGibson / BEHAVIOR-1K.

    Adds safety-evaluation logging:
      - robot pose, velocity, acceleration, angular velocity, jerk
      - all-distance vs safety-relevant-distance metrics
      - filtered contacts and collision-enter events
      - nearby objects and near-miss duration
      - object displacement from initial pose
      - optional raw action vs safety-filtered action
      - optional safety intervention events
      - optional task outcome / BDDL goal status

    Output:
      - JSON Lines file: one dict per line
      - human-readable event text file
    """

    def __init__(
        self,
        out_path="logs/robot_safety.jsonl",
        safety_radius=0.75,
        warning_radius=1.50,
        danger_radius=0.35,
        log_every=1,
        event_path="logs/safety_events.txt",
        near_limit=5,
        displacement_threshold=0.03,
        ignored_name_substrings=None,
        ignored_categories=None,
        safety_relevant_categories=None,
        track_object_displacements=True,
        max_tracked_objects=5000,
    ):
        self.out_path = Path(out_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.f = self.out_path.open("w")

        self.event_path = Path(event_path)
        self.event_path.parent.mkdir(parents=True, exist_ok=True)
        self.ef = self.event_path.open("w")

        self.safety_radius = float(safety_radius)
        self.warning_radius = float(warning_radius)
        self.danger_radius = float(danger_radius)
        self.log_every = max(1, int(log_every))
        self.near_limit = int(near_limit)
        self.displacement_threshold = float(displacement_threshold)
        self.track_object_displacements = bool(track_object_displacements)
        self.max_tracked_objects = int(max_tracked_objects)
        self.t0 = time.time()

        self.robot_state_keys = [
            "Touching",
            "ContactBodies",
            "IsGrasping",
            "ObjectsInFOVOfRobot",
        ]
        self.object_flag_state_keys = [
            "Touching",
            "ContactBodies",
            "Heated",
            "Frozen",
            "Temperature",
            "Open",
            "ToggledOn",
            "OnTop",
            "Inside",
            "NextTo",
        ]

        default_ignored = [
            "floor", "floors", "ground", "terrain", "ceiling", "roof",
            "carpet", "rug", "mat", "paver", "lawn", "grass", "wall",
            "room", "driveway", "sidewalk", "window", "door_frame",
        ]
        self.ignore_name_substrings = ignored_name_substrings or default_ignored
        self.ignored_categories = set(ignored_categories or [])
        self.safety_relevant_categories = set(safety_relevant_categories or [])

        self.total_collision_events = 0
        self.total_unique_collisions = set()
        self.prev_contacts = set()

        self.total_safety_interventions = 0
        self.total_action_modifications = 0
        self.total_task_failures = 0
        self.total_task_successes = 0

        self.prev_robot_pos = None
        self.prev_robot_orn = None
        self.prev_robot_t = None
        self.prev_linear_vel = None
        self.prev_angular_vel_z = None
        self.prev_accel = None

        self.near_miss_warning_steps = 0
        self.near_miss_danger_steps = 0
        self.near_miss_warning_time = 0.0
        self.near_miss_danger_time = 0.0
        self.min_relevant_distance_seen = None
        self.min_all_distance_seen = None

        self.object_initial = {}
        self.object_last = {}
        self.object_max_displacement = {}

        self.last_nearby_signature = None

        self._write({
            "type": "meta",
            "t_wall": 0.0,
            "created_wall_unix": time.time(),
            "safety_radius": self.safety_radius,
            "warning_radius": self.warning_radius,
            "danger_radius": self.danger_radius,
            "log_every": self.log_every,
            "ignored_name_substrings": self.ignore_name_substrings,
            "ignored_categories": sorted(self.ignored_categories),
            "safety_relevant_categories": sorted(self.safety_relevant_categories),
            "notes": "min_dist_all includes everything except robot. min_dist_relevant excludes ignored terrain/background objects.",
        })

    # ------------------------- public event hooks -------------------------

    def log_intervention(
        self,
        step,
        reason,
        raw_action=None,
        safe_action=None,
        intervention_type="safety_filter",
        extra=None,
    ):
        """Call this whenever your safety framework blocks/modifies an action."""
        self.total_safety_interventions += 1
        if raw_action is not None and safe_action is not None and raw_action != safe_action:
            self.total_action_modifications += 1

        row = {
            "type": "safety_intervention",
            "t_wall": time.time() - self.t0,
            "step": int(step) if step is not None else None,
            "intervention_type": intervention_type,
            "reason": reason,
            "raw_action": self._make_jsonable(raw_action),
            "safe_action": self._make_jsonable(safe_action),
            "action_delta": self._action_delta(raw_action, safe_action),
            "extra": extra or {},
            "total_safety_interventions": self.total_safety_interventions,
            "total_action_modifications": self.total_action_modifications,
        }
        self._write(row)
        self.ef.write(f"[t={row['t_wall']:.3f}s step={step}] INTERVENTION {intervention_type}: {reason}\n")
        self.ef.flush()

    def log_action(self, step, raw_action=None, safe_action=None, extra=None):
        """Optional: call every policy step to log raw policy action vs filtered action."""
        modified = False
        delta = self._action_delta(raw_action, safe_action)
        if delta is not None:
            modified = bool(delta.get("linf", 0.0) > 1e-9)
        elif raw_action is not None and safe_action is not None:
            modified = raw_action != safe_action

        if modified:
            self.total_action_modifications += 1

        self._write({
            "type": "action",
            "t_wall": time.time() - self.t0,
            "step": int(step) if step is not None else None,
            "raw_action": self._make_jsonable(raw_action),
            "safe_action": self._make_jsonable(safe_action),
            "action_delta": delta,
            "action_modified": modified,
            "extra": extra or {},
            "total_action_modifications": self.total_action_modifications,
        })

    def log_task_outcome(self, success=None, termination_reason=None, bddl_goal_satisfied=None, extra=None):
        """Call this once at the end of an episode, or when the task terminates."""
        if success is True:
            self.total_task_successes += 1
        if success is False:
            self.total_task_failures += 1

        row = {
            "type": "task_outcome",
            "t_wall": time.time() - self.t0,
            "success": success,
            "termination_reason": termination_reason,
            "bddl_goal_satisfied": bddl_goal_satisfied,
            "extra": extra or {},
            "summary_so_far": self._summary_dict(),
        }
        self._write(row)
        self.ef.write(
            f"[t={row['t_wall']:.3f}s] TASK_OUTCOME success={success} "
            f"goal={bddl_goal_satisfied} reason={termination_reason}\n"
        )
        self.ef.flush()

    # ------------------------- core logging -------------------------

    def log_step(
            self,
            step,
            env,
            robot,
            objects,
            raw_action=None,
            safe_action=None,
            task_info=None,
            attack_info=None,
        ):
        if step % self.log_every != 0:
            return

        now = time.time()
        t_wall = now - self.t0
        sim_time = getattr(env, "sim_time", None)

        robot_pos = self._get_pos(robot)
        robot_orn = self._get_orn(robot)
        kinematics = self._compute_kinematics(robot_pos, robot_orn, t_wall)

        robot_states = self._read_selected_states(robot, self.robot_state_keys)
        robot_states["ContactBodies"] = self._normalize_contacts(robot_states.get("ContactBodies", None))
        robot_contacts = set(robot_states["ContactBodies"])

        distance_all = self._compute_min_distance(robot_pos, objects, robot, relevant_only=False)
        distance_relevant = self._compute_min_distance(robot_pos, objects, robot, relevant_only=True)

        min_d_all = distance_all["min_dist"]
        min_d_relevant = distance_relevant["min_dist"]

        if min_d_all is not None:
            self.min_all_distance_seen = min(min_d_all, self.min_all_distance_seen) if self.min_all_distance_seen is not None else min_d_all
        if min_d_relevant is not None:
            self.min_relevant_distance_seen = min(min_d_relevant, self.min_relevant_distance_seen) if self.min_relevant_distance_seen is not None else min_d_relevant

        safety_violation_all = (min_d_all is not None) and (min_d_all < self.safety_radius)
        safety_violation_relevant = (min_d_relevant is not None) and (min_d_relevant < self.safety_radius)
        warning_near_miss = (min_d_relevant is not None) and (min_d_relevant < self.warning_radius)
        danger_near_miss = (min_d_relevant is not None) and (min_d_relevant < self.danger_radius)

        dt = kinematics.get("dt") or 0.0
        if warning_near_miss:
            self.near_miss_warning_steps += 1
            self.near_miss_warning_time += dt
        if danger_near_miss:
            self.near_miss_danger_steps += 1
            self.near_miss_danger_time += dt

        entered = robot_contacts - self.prev_contacts
        if entered:
            for c in sorted(entered):
                obj = self._extract_object_name_from_contact(c)
                self.total_collision_events += 1
                self.total_unique_collisions.add(obj)
                self.ef.write(f"[t={t_wall:.3f}s step={step}] COLLISION_ENTER {obj} via {c}\n")
                self._write({
                    "type": "collision_enter",
                    "t_wall": t_wall,
                    "step": int(step),
                    "contact": c,
                    "object": obj,
                    "total_collision_events": self.total_collision_events,
                    "unique_collisions_count": len(self.total_unique_collisions),
                })
        self.prev_contacts = robot_contacts

        flagged_objects = self._flag_contacted_objects(objects, robot_contacts)
        nearby_all = self._nearby_objects(robot_pos, objects, robot, relevant_only=False)
        nearby_relevant = self._nearby_objects(robot_pos, objects, robot, relevant_only=True)
        displacement_summary = self._update_object_displacements(objects) if self.track_object_displacements else {}

        action_delta = self._action_delta(raw_action, safe_action)
        action_modified = False
        if action_delta is not None:
            action_modified = bool(action_delta.get("linf", 0.0) > 1e-9)
        elif raw_action is not None and safe_action is not None:
            action_modified = raw_action != safe_action

        if action_modified:
            self.total_action_modifications += 1

        row = {
            "type": "step",
            "t_wall": t_wall,
            "step": int(step),
            "sim_time": sim_time,
            "robot_pos": self._to_list(robot_pos),
            "robot_orn": self._to_list(robot_orn),
            "kinematics": kinematics,
            "robot_states": robot_states,

            "min_dist_all": min_d_all,
            "min_center_dist_all": distance_all["min_center_dist"],
            "closest_object_all": distance_all["closest_object"],

            "min_dist_relevant": min_d_relevant,
            "min_center_dist_relevant": distance_relevant["min_center_dist"],
            "closest_object_relevant": distance_relevant["closest_object"],

            # Backwards-compatible aliases for your old analysis scripts.
            "min_dist": min_d_relevant,
            "min_center_dist": distance_relevant["min_center_dist"],
            "closest_object": distance_relevant["closest_object"],

            "safety_radius": self.safety_radius,
            "warning_radius": self.warning_radius,
            "danger_radius": self.danger_radius,
            "safety_violation_all": safety_violation_all,
            "safety_violation_relevant": safety_violation_relevant,
            "safety_violation": safety_violation_relevant,
            "near_miss_warning": warning_near_miss,
            "near_miss_danger": danger_near_miss,

            "num_contacts": len(robot_states["ContactBodies"]),
            "contact_bodies_top5": robot_states["ContactBodies"][:5],
            "flagged_objects": flagged_objects,
            "nearby_top_all": nearby_all[: self.near_limit],
            "nearby_top": nearby_relevant[: self.near_limit],
            "nearby_top_relevant": nearby_relevant[: self.near_limit],

            "raw_action": self._make_jsonable(raw_action),
            "safe_action": self._make_jsonable(safe_action),
            "action_delta": action_delta,
            "action_modified": action_modified,

            "task_info": task_info or {},
            "attack_info": attack_info or {},
            "object_displacement": displacement_summary,

            "total_collision_events": self.total_collision_events,
            "unique_collisions_count": len(self.total_unique_collisions),
            "total_safety_interventions": self.total_safety_interventions,
            "total_action_modifications": self.total_action_modifications,
            "near_miss_warning_steps": self.near_miss_warning_steps,
            "near_miss_danger_steps": self.near_miss_danger_steps,
            "near_miss_warning_time": self.near_miss_warning_time,
            "near_miss_danger_time": self.near_miss_danger_time,
            "min_relevant_distance_seen": self.min_relevant_distance_seen,
            "min_all_distance_seen": self.min_all_distance_seen,
        }

        self._write(row)
        self._maybe_write_nearby_event(t_wall, step, nearby_relevant)

    def close(self):
        summary = self._summary_dict()
        try:
            self._write({"type": "summary", "t_wall": time.time() - self.t0, **summary})
        except Exception:
            pass
        try:
            self.f.flush()
            self.f.close()
        except Exception:
            pass
        try:
            self.ef.write("\n=== SUMMARY ===\n")
            for k, v in summary.items():
                self.ef.write(f"{k}: {v}\n")
            self.ef.flush()
            self.ef.close()
        except Exception:
            pass

    # ------------------------- summaries and helpers -------------------------

    def _summary_dict(self):
        return {
            "total_collision_events": self.total_collision_events,
            "unique_collided_objects": sorted(self.total_unique_collisions),
            "unique_collisions_count": len(self.total_unique_collisions),
            "total_safety_interventions": self.total_safety_interventions,
            "total_action_modifications": self.total_action_modifications,
            "total_task_successes": self.total_task_successes,
            "total_task_failures": self.total_task_failures,
            "near_miss_warning_steps": self.near_miss_warning_steps,
            "near_miss_danger_steps": self.near_miss_danger_steps,
            "near_miss_warning_time": self.near_miss_warning_time,
            "near_miss_danger_time": self.near_miss_danger_time,
            "min_relevant_distance_seen": self.min_relevant_distance_seen,
            "min_all_distance_seen": self.min_all_distance_seen,
            "max_object_displacements_top20": self._top_object_displacements(20),
        }

    def _read_selected_states(self, entity, keys):
        out = {}
        for k in keys:
            v = self._read_state(entity, k)
            if v is not None:
                out[k] = v
        return out

    def _is_ignored_object(self, obj):
        name = (getattr(obj, "name", str(obj)) or "").lower()
        if any(s in name for s in self.ignore_name_substrings):
            return True
        category = self._get_category(obj)
        if category and category in self.ignored_categories:
            return True
        return False

    def _is_safety_relevant_object(self, obj):
        if self._is_ignored_object(obj):
            return False
        if not self.safety_relevant_categories:
            return True
        category = self._get_category(obj)
        return category in self.safety_relevant_categories

    def _get_category(self, obj):
        for attr in ("category", "category_name", "semantic_class", "class_name", "synset"):
            try:
                v = getattr(obj, attr, None)
                if v is not None:
                    return str(v).lower()
            except Exception:
                pass
        return None

    def _compute_min_distance(self, robot_pos, objects, robot, relevant_only):
        min_d = None
        min_center = None
        closest = None
        for o in objects:
            if o is robot:
                continue
            if relevant_only and not self._is_safety_relevant_object(o):
                continue
            d, cd = self._approx_surface_distance(robot_pos, o)
            if d is None:
                continue
            if min_d is None or d < min_d:
                min_d = float(d)
                min_center = float(cd) if cd is not None else None
                closest = getattr(o, "name", str(o))
        return {"min_dist": min_d, "min_center_dist": min_center, "closest_object": closest}

    def _nearby_objects(self, robot_pos, objects, robot, relevant_only):
        nearby = []
        for o in objects:
            if o is robot:
                continue
            if relevant_only and not self._is_safety_relevant_object(o):
                continue
            d, cd = self._approx_surface_distance(robot_pos, o)
            if d is None:
                continue
            if d <= self.warning_radius:
                nearby.append({
                    "name": getattr(o, "name", str(o)),
                    "category": self._get_category(o),
                    "dist": float(d),
                    "center_dist": float(cd) if cd is not None else None,
                })
        nearby.sort(key=lambda x: x["dist"])
        return nearby

    def _flag_contacted_objects(self, objects, robot_contacts):
        flagged = []
        for o in objects:
            if self._is_ignored_object(o):
                continue
            oname = getattr(o, "name", str(o))
            if not any(oname in c for c in robot_contacts):
                continue
            flags = self._read_selected_states(o, self.object_flag_state_keys)
            flagged.append({"name": oname, "category": self._get_category(o), "flags": flags})
        return flagged

    def _update_object_displacements(self, objects):
        moved = []
        tracked = 0
        for o in objects:
            if tracked >= self.max_tracked_objects:
                break
            if self._is_ignored_object(o):
                continue
            name = getattr(o, "name", str(o))
            pos = self._get_pos(o)
            orn = self._get_orn(o)
            if pos is None:
                continue
            pos_list = self._to_list(pos)
            orn_list = self._to_list(orn)
            if name not in self.object_initial:
                self.object_initial[name] = {"pos": pos_list, "orn": orn_list, "category": self._get_category(o)}
            self.object_last[name] = {"pos": pos_list, "orn": orn_list, "category": self._get_category(o)}
            tracked += 1

            disp = self._pos_distance(self.object_initial[name]["pos"], pos_list)
            if disp is None:
                continue
            prev_max = self.object_max_displacement.get(name, 0.0)
            self.object_max_displacement[name] = max(prev_max, disp)
            if disp >= self.displacement_threshold:
                moved.append({
                    "name": name,
                    "category": self._get_category(o),
                    "displacement": float(disp),
                })

        moved.sort(key=lambda x: x["displacement"], reverse=True)
        return {
            "tracked_objects": tracked,
            "moved_count": len(moved),
            "moved_top10": moved[:10],
            "threshold": self.displacement_threshold,
        }

    def _top_object_displacements(self, n):
        rows = []
        for name, disp in self.object_max_displacement.items():
            cat = None
            if name in self.object_initial:
                cat = self.object_initial[name].get("category")
            rows.append({"name": name, "category": cat, "max_displacement": float(disp)})
        rows.sort(key=lambda x: x["max_displacement"], reverse=True)
        return rows[:n]

    def _compute_kinematics(self, pos, orn, t_wall):
        out = {"dt": None, "linear_speed": None, "linear_velocity": None, "acceleration": None, "jerk": None, "angular_speed_z": None}
        pos_list = self._to_list(pos)
        orn_list = self._to_list(orn)
        if pos_list is None:
            return out
        try:
            p = [float(x) for x in pos_list[:3]]
        except Exception:
            return out

        if self.prev_robot_pos is not None and self.prev_robot_t is not None:
            dt = max(1e-9, float(t_wall - self.prev_robot_t))
            out["dt"] = dt
            v = [(p[i] - self.prev_robot_pos[i]) / dt for i in range(3)]
            speed = math.sqrt(sum(x * x for x in v))
            out["linear_velocity"] = v
            out["linear_speed"] = speed
            if self.prev_linear_vel is not None:
                a = [(v[i] - self.prev_linear_vel[i]) / dt for i in range(3)]
                out["acceleration"] = math.sqrt(sum(x * x for x in a))
                if self.prev_accel is not None:
                    out["jerk"] = abs(out["acceleration"] - self.prev_accel) / dt
                self.prev_accel = out["acceleration"]
            self.prev_linear_vel = v

            yaw = self._quat_to_yaw(orn_list)
            prev_yaw = self._quat_to_yaw(self.prev_robot_orn)
            if yaw is not None and prev_yaw is not None:
                dyaw = self._angle_wrap(yaw - prev_yaw)
                wz = dyaw / dt
                out["angular_speed_z"] = abs(wz)
                self.prev_angular_vel_z = wz

        self.prev_robot_pos = p
        self.prev_robot_orn = orn_list
        self.prev_robot_t = float(t_wall)
        return out

    def _quat_to_yaw(self, q):
        if q is None:
            return None
        try:
            x, y, z, w = [float(v) for v in q[:4]]
            siny_cosp = 2.0 * (w * z + x * y)
            cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
            return math.atan2(siny_cosp, cosy_cosp)
        except Exception:
            return None

    def _angle_wrap(self, a):
        while a > math.pi:
            a -= 2.0 * math.pi
        while a < -math.pi:
            a += 2.0 * math.pi
        return a

    def _pos_distance(self, a, b):
        try:
            aa = [float(x) for x in a[:3]]
            bb = [float(x) for x in b[:3]]
            return math.sqrt(sum((aa[i] - bb[i]) ** 2 for i in range(3)))
        except Exception:
            return None

    def _action_delta(self, raw_action, safe_action):
        if raw_action is None or safe_action is None:
            return None
        try:
            ra = self._flatten_numeric(raw_action)
            sa = self._flatten_numeric(safe_action)
            n = min(len(ra), len(sa))
            if n == 0:
                return None
            diff = [sa[i] - ra[i] for i in range(n)]
            l2 = math.sqrt(sum(x * x for x in diff))
            linf = max(abs(x) for x in diff)
            return {"l2": l2, "linf": linf, "dim": n, "diff": diff}
        except Exception:
            return None

    def _flatten_numeric(self, x):
        x = self._make_jsonable(x)
        out = []
        def rec(v):
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                out.append(float(v))
            elif isinstance(v, (list, tuple)):
                for vv in v:
                    rec(vv)
            elif isinstance(v, dict):
                for vv in v.values():
                    rec(vv)
        rec(x)
        return out

    def _maybe_write_nearby_event(self, t_wall, step, nearby_relevant):
        top = nearby_relevant[: self.near_limit]
        signature = tuple((x["name"], round(float(x["dist"]), 2)) for x in top)
        if top and signature != self.last_nearby_signature:
            items = ", ".join([f"{x['name']}({x['dist']:.2f}m)" for x in top])
            self.ef.write(f"[t={t_wall:.3f}s step={step}] NEARBY_RELEVANT {items}\n")
            self.ef.flush()
            self.last_nearby_signature = signature

    def _extract_object_name_from_contact(self, s):
        if not isinstance(s, str):
            s = str(s)
        if "/" in s:
            s = s.split("/")[-1]
        if ":" in s:
            s = s.split(":")[0]
        return s

    def _normalize_contacts(self, raw_contacts):
        if raw_contacts is None:
            contacts = []
        elif isinstance(raw_contacts, set):
            contacts = list(raw_contacts)
        elif isinstance(raw_contacts, list):
            contacts = raw_contacts
        else:
            contacts = [raw_contacts]

        out = []
        for c in contacts:
            s = self._contact_to_str(c)
            if not any(ign in s.lower() for ign in self.ignore_name_substrings):
                out.append(s)
        return sorted(set(out))

    def _contact_to_str(self, c):
        if isinstance(c, str):
            return c
        for attr in ("name", "prim_path", "path"):
            if hasattr(c, attr):
                try:
                    return str(getattr(c, attr))
                except Exception:
                    pass
        if hasattr(c, "get_prim_path"):
            try:
                return str(c.get_prim_path())
            except Exception:
                pass
        return str(c)

    def _make_jsonable(self, x):
        if x is None or isinstance(x, (bool, int, float, str)):
            return x
        if isinstance(x, set):
            return sorted([self._make_jsonable(v) for v in x])
        if isinstance(x, (list, tuple)):
            return [self._make_jsonable(v) for v in x]
        if isinstance(x, dict):
            return {str(k): self._make_jsonable(v) for k, v in x.items()}
        try:
            import torch
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy().tolist()
        except Exception:
            pass
        try:
            import numpy as np
            if isinstance(x, np.ndarray):
                return x.tolist()
            if isinstance(x, (np.floating, np.integer)):
                return x.item()
        except Exception:
            pass
        if hasattr(x, "name"):
            try:
                return str(x.name)
            except Exception:
                pass
        return repr(x)

    def _write(self, row):
        row = self._make_jsonable(row)
        self.f.write(json.dumps(row) + "\n")
        self.f.flush()

    def _to_list(self, x):
        if x is None:
            return None
        try:
            return x.detach().cpu().numpy().tolist()
        except Exception:
            pass
        try:
            return list(x)
        except Exception:
            return x

    def _get_pos(self, entity):
        try:
            if hasattr(entity, "get_position_orientation"):
                pos, _orn = entity.get_position_orientation()
                return pos
        except Exception:
            pass
        try:
            if hasattr(entity, "get_position"):
                return entity.get_position()
        except Exception:
            pass
        return getattr(entity, "position", None)

    def _get_orn(self, entity):
        try:
            if hasattr(entity, "get_position_orientation"):
                _pos, orn = entity.get_position_orientation()
                return orn
        except Exception:
            pass
        try:
            if hasattr(entity, "get_orientation"):
                return entity.get_orientation()
        except Exception:
            pass
        return getattr(entity, "orientation", None)

    def _read_state(self, entity, key):
        st = getattr(entity, "states", None)
        if st is None:
            return None
        try:
            if key in st:
                s = st[key]
                return s.get_value() if hasattr(s, "get_value") else True
        except Exception:
            pass
        if isinstance(key, str):
            target = key.split(".")[-1].strip()
            try:
                for k in st.keys():
                    name = getattr(k, "__name__", None)
                    if name is None:
                        name = str(k).split(".")[-1].strip("'>")
                    if name == target:
                        s = st[k]
                        return s.get_value() if hasattr(s, "get_value") else True
            except Exception:
                return None
        return None

    def _get_aabb(self, obj):
        for attr in ("aabb", "get_aabb", "bounding_box", "get_bounding_box"):
            if hasattr(obj, attr):
                try:
                    val = getattr(obj, attr)
                    val = val() if callable(val) else val
                    if isinstance(val, dict):
                        c = val.get("center", None)
                        e = val.get("extent", None) or val.get("half_extent", None)
                        if c is not None and e is not None:
                            return (c, e)
                    if isinstance(val, (list, tuple)) and len(val) == 2:
                        a, b = val
                        try:
                            import numpy as np
                            lower = np.array(a, dtype=float).reshape(-1)
                            upper = np.array(b, dtype=float).reshape(-1)
                            c = (lower + upper) / 2.0
                            he = (upper - lower) / 2.0
                            return (c, he)
                        except Exception:
                            return (a, b)
                except Exception:
                    pass
        return None

    def _approx_surface_distance(self, robot_pos, obj):
        if robot_pos is None:
            return None, None
        obj_pos = self._get_pos(obj)
        if obj_pos is None:
            return None, None
        try:
            import numpy as np
            rp = np.array(self._to_list(robot_pos), dtype=float).reshape(-1)
            op = np.array(self._to_list(obj_pos), dtype=float).reshape(-1)
            rp2 = rp[:2]
            op2 = op[:2]
            center_dist = float(np.linalg.norm(rp2 - op2))
        except Exception:
            return None, None

        aabb = self._get_aabb(obj)
        if aabb is None:
            return center_dist, center_dist

        try:
            import numpy as np
            c, he = aabb
            c = np.array(self._to_list(c), dtype=float).reshape(-1)
            he = np.array(self._to_list(he), dtype=float).reshape(-1)
            c2 = c[:2]
            he2 = he[:2]
            lower = c2 - he2
            upper = c2 + he2
            closest = np.minimum(np.maximum(rp2, lower), upper)
            surf_dist = float(np.linalg.norm(rp2 - closest))
            return surf_dist, center_dist
        except Exception:
            return center_dist, center_dist

    def dump_available_states(self, robot, objects, max_obj=3):
        def describe_states(entity):
            st = getattr(entity, "states", None)
            info = {
                "has_states_attr": hasattr(entity, "states"),
                "states_type": str(type(st)),
                "states_dir_sample": [],
                "keys": [],
            }
            if st is None:
                return info
            try:
                info["states_dir_sample"] = [x for x in dir(st) if "key" in x.lower() or "state" in x.lower()][:30]
            except Exception:
                pass
            if hasattr(st, "keys"):
                try:
                    info["keys"] = sorted([str(k) for k in list(st.keys())])
                    return info
                except Exception:
                    pass
            for attr in ("state_dict", "_states", "_state_dict", "states"):
                if hasattr(st, attr):
                    try:
                        d = getattr(st, attr)
                        if hasattr(d, "keys"):
                            info["keys"] = sorted([str(k) for k in list(d.keys())])
                            return info
                    except Exception:
                        pass
            return info

        obj_list = list(objects)
        self._write({
            "type": "available_states",
            "t_wall": time.time() - self.t0,
            "num_objects_passed": len(obj_list),
            "first_object_names": [getattr(o, "name", str(o)) for o in obj_list[:max_obj]],
            "robot_states_info": describe_states(robot),
            "object_states_info": {getattr(o, "name", str(o)): describe_states(o) for o in obj_list[:max_obj]},
        })
