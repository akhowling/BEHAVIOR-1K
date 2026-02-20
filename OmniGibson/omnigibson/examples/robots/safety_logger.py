# ~/BEHAVIOR-1K/safety_logger.py
import json
import time
from pathlib import Path

class SafetyLogger:
    """
    JSONL safety logger for OmniGibson / BEHAVIOR-1K.

    Logs per step:
      - robot pose (pos, orn if available)
      - selected semantic states (whatever exists on your version)
      - flagged objects with states like OnFire / InCollision / InContact (if present)
      - approximate min distance to objects (AABB if available, else center distance)
      - safety radius violation

    Output: JSON Lines (.jsonl): one dict per line.
    """

    # def __init__(self, out_path="logs/robot_safety.jsonl", safety_radius=0.75, log_every=1):
    def __init__(self, out_path="logs/robot_safety.jsonl", safety_radius=0.75, log_every=1,
             event_path="logs/safety_events.txt", near_limit=5):
        self.out_path = Path(out_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.f = self.out_path.open("w")
        self.safety_radius = float(safety_radius)
        self.log_every = int(log_every)
        self.t0 = time.time()

        # Candidate state keys (names vary by OG version; we only log those that exist)
        self.robot_state_keys = ["Touching", "ContactBodies", "IsGrasping", "ObjectsInFOVOfRobot"]
        self.object_flag_state_keys = ["Touching", "ContactBodies", "Heated", "Frozen", "Temperature"]

        self.last_nearby_signature = None
        self.last_nearby_line = None




        #ignore floors and stuff
        self.ignore_name_substrings = ["floor", "floors", "ground", "terrain", "ceiling","carpet", "rug", "mat",]

        #humanlogs
        self.event_path = Path(event_path)
        self.event_path.parent.mkdir(parents=True, exist_ok=True)
        self.ef = self.event_path.open("w")   # human readable
        self.near_limit = int(near_limit)

        self.total_collision_events = 0
        self.total_unique_collisions = set()  # object names
        self.prev_contacts = set()


        self._write({
            "type": "meta",
            "t_wall": 0.0,
            "created_wall_unix": time.time(),
            "safety_radius": self.safety_radius,
        })

    def close(self):
        try:
            self.f.flush(); self.f.close()
        except Exception:
            pass
        try:
            # write a final summary
            self.ef.write("\n=== SUMMARY ===\n")
            self.ef.write(f"total collision events: {self.total_collision_events}\n")
            self.ef.write(f"unique collided objects: {sorted(self.total_unique_collisions)}\n")
            self.ef.flush(); self.ef.close()
        except Exception:
            pass


    # def _write(self, row):
    #     self.f.write(json.dumps(row) + "\n")
    #     self.f.flush()

    def _extract_object_name_from_contact(self, s: str) -> str:
        # examples: "coffee_table_x:base_link", "/World/.../coffee_table_x/base_link"
        if not isinstance(s, str):
            s = str(s)
        # take last path segment if USD path
        if "/" in s:
            s = s.split("/")[-1]
        # drop link suffix after colon
        if ":" in s:
            s = s.split(":")[0]
        return s


    def _make_jsonable(self, x):
        # basic primitives
        if x is None or isinstance(x, (bool, int, float, str)):
            return x

        # sets -> sorted list (stable)
        if isinstance(x, set):
            return sorted([self._make_jsonable(v) for v in x])

        # tuples / lists
        if isinstance(x, (list, tuple)):
            return [self._make_jsonable(v) for v in x]

        # dicts
        if isinstance(x, dict):
            # JSON requires string keys
            out = {}
            for k, v in x.items():
                out[str(k)] = self._make_jsonable(v)
            return out

        # torch tensors
        try:
            import torch
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy().tolist()
        except Exception:
            pass

        # numpy arrays / scalars
        try:
            import numpy as np
            if isinstance(x, np.ndarray):
                return x.tolist()
            if isinstance(x, (np.floating, np.integer)):
                return x.item()
        except Exception:
            pass

        # OmniGibson / Isaac objects: try a name, else repr
        if hasattr(x, "name"):
            try:
                return str(x.name)
            except Exception:
                pass

        # last resort
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
        # Preferred new API
        try:
            if hasattr(entity, "get_position_orientation"):
                pos, _orn = entity.get_position_orientation()
                return pos
        except Exception:
            pass

        # Deprecated but still works
        try:
            if hasattr(entity, "get_position"):
                return entity.get_position()
        except Exception:
            pass

        return getattr(entity, "position", None)


    def _get_orn(self, entity):
        # Preferred new API
        try:
            if hasattr(entity, "get_position_orientation"):
                _pos, orn = entity.get_position_orientation()
                return orn
        except Exception:
            pass

        # Deprecated fallback
        try:
            if hasattr(entity, "get_orientation"):
                return entity.get_orientation()
        except Exception:
            pass

        return None


    def _read_state(self, entity, key):
        """
        Reads an OmniGibson object state from entity.states.

        Supports states dict keyed by:
        - state classes (common in OG): {Touching: TouchingStateInstance, ...}
        - strings (less common): {"Touching": ...}

        `key` can be:
        - a class (e.g. Touching)
        - a string class name (e.g. "Touching")
        - a fully-qualified string that ends with the class name
        """
        st = getattr(entity, "states", None)
        if st is None:
            return None

        try:
            # Direct lookup if user passed the actual key type
            if key in st:
                s = st[key]
                if hasattr(s, "get_value"):
                    return s.get_value()
                return True
        except Exception:
            pass

        # If key is a string, try to match against class keys by name
        if isinstance(key, str):
            target = key.split(".")[-1].strip()  # allow "Touching" or "....Touching"
            try:
                for k in st.keys():
                    # k might be a class
                    name = getattr(k, "__name__", None)
                    if name is None:
                        # fallback string representation
                        name = str(k).split(".")[-1].strip("'>")
                    if name == target:
                        s = st[k]
                        if hasattr(s, "get_value"):
                            return s.get_value()
                        return True
            except Exception:
                return None

        return None


    def _get_aabb(self, obj):
        """
        Attempts to extract an AABB from common OG attributes.
        Returns (center, half_extents) if found, else None.
        """
        for attr in ("aabb", "get_aabb", "bounding_box", "get_bounding_box"):
            if hasattr(obj, attr):
                try:
                    val = getattr(obj, attr)
                    val = val() if callable(val) else val

                    # dict format
                    if isinstance(val, dict):
                        c = val.get("center", None)
                        e = val.get("extent", None) or val.get("half_extent", None)
                        if c is not None and e is not None:
                            return (c, e)

                    # tuple/list formats
                    if isinstance(val, (list, tuple)) and len(val) == 2:
                        a, b = val
                        # assume (lower, upper) if possible
                        try:
                            import numpy as np
                            lower = np.array(a, dtype=float).reshape(-1)
                            upper = np.array(b, dtype=float).reshape(-1)
                            c = (lower + upper) / 2.0
                            he = (upper - lower) / 2.0
                            return (c, he)
                        except Exception:
                            # maybe already (center, half_extents)
                            return (a, b)
                except Exception:
                    pass
        return None

    def _approx_surface_distance(self, robot_pos, obj):
        """
        Distance from robot point to object surface using AABB if available,
        else center distance.
        Returns (surface_or_center_dist, center_dist).
        """
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
            # center_dist = float(np.linalg.norm(rp - op))
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
            # lower = c - he
            # upper = c + he
            # closest = np.minimum(np.maximum(rp, lower), upper)
            # surf_dist = float(np.linalg.norm(rp - closest))

            c2 = c[:2]
            he2 = he[:2]
            lower = c2 - he2
            upper = c2 + he2
            closest = np.minimum(np.maximum(rp2, lower), upper)
            surf_dist = float(np.linalg.norm(rp2 - closest))

            return surf_dist, center_dist
        except Exception:
            return center_dist, center_dist

    def log_step(self, step, env, robot, objects):
        if step % self.log_every != 0:
            return

        robot_pos = self._get_pos(robot)
        robot_orn = self._get_orn(robot)

        # Robot semantic states (only those that exist)
        robot_states = {}
        for k in self.robot_state_keys:
            v = self._read_state(robot, k)
            if v is not None:
                robot_states[k] = v

        # robot_contacts = robot_states.get("ContactBodies", [])
        # if robot_contacts is None:
        #     robot_contacts = []
        # robot_contacts = set(robot_contacts)

        # if "ContactBodies" in robot_states and isinstance(robot_states["ContactBodies"], list):
        #     robot_states["ContactBodies"] = [
        #         c for c in robot_states["ContactBodies"]
        #         if not any(s in c.lower() for s in self.ignore_name_substrings)
        #     ]


        # --- Normalize and filter robot contacts ---
        raw_contacts = robot_states.get("ContactBodies", None)

        # Normalize to a flat list
        if raw_contacts is None:
            contacts = []
        elif isinstance(raw_contacts, set):
            contacts = list(raw_contacts)
        elif isinstance(raw_contacts, list):
            contacts = raw_contacts
        else:
            contacts = [raw_contacts]

        def contact_to_str(c):
            # Already string?
            if isinstance(c, str):
                return c
            # Common OG / Isaac prims have name or prim_path
            for attr in ("name", "prim_path", "path"):
                if hasattr(c, attr):
                    try:
                        return str(getattr(c, attr))
                    except Exception:
                        pass
            # Some prims have get_prim_path()
            if hasattr(c, "get_prim_path"):
                try:
                    return str(c.get_prim_path())
                except Exception:
                    pass
            # Fallback
            return str(c)

        contacts_str = [contact_to_str(c) for c in contacts]

        # Filter out floor / walls / carpets etc.
        contacts_str = [
            s for s in contacts_str
            if not any(ign in s.lower() for ign in self.ignore_name_substrings)
        ]

        robot_states["ContactBodies"] = contacts_str
        robot_contacts = set(contacts_str)




        # Flag objects with interesting states (skip floors etc. here too if you want)
        # flagged_objects = []
        # for o in objects:
        #     oname = (getattr(o, "name", "") or "").lower()
        #     if any(s in oname for s in self.ignore_name_substrings):
        #         continue

        #     flags = {}
        #     for k in self.object_flag_state_keys:
        #         v = self._read_state(o, k)
        #         if v is not None:
        #             flags[k] = v
        #     if flags:
        #         flagged_objects.append({"name": getattr(o, "name", str(o)), "flags": flags})
        flagged_objects = []

        for o in objects:
            oname = getattr(o, "name", str(o))
            lname = oname.lower()
            if any(s in lname for s in self.ignore_name_substrings):
                continue

            # Only log objects that robot is contacting
            # Many OG objects have ContactBodies listing who they contact; robot lists contacts too.
            # We'll check if this object's base_link appears in robot_contacts OR vice versa.
            # obj_id = f"{oname}:base_link"

            # if (obj_id not in robot_contacts):
            #     continue

            if not any(oname in c for c in robot_contacts):
                continue


            flags = {}
            for k in self.object_flag_state_keys:
                v = self._read_state(o, k)
                if v is not None:
                    flags[k] = v

            flagged_objects.append({"name": oname, "flags": flags})


        # Min distance to objects (approx)
        min_d = None
        closest = None
        min_center = None

        for o in objects:
            if o is robot:
                continue

            oname = (getattr(o, "name", "") or "").lower()
            if any(s in oname for s in self.ignore_name_substrings):
                continue

            d, cd = self._approx_surface_distance(robot_pos, o)
            if d is None:
                continue
            if (min_d is None) or (d < min_d):
                min_d = float(d)
                min_center = float(cd) if cd is not None else None
                closest = getattr(o, "name", str(o))

        safety_violation = (min_d is not None) and (min_d < self.safety_radius)

        # --- Collision events (edge-triggered) ---
        current_contacts = set(robot_states.get("ContactBodies", []))  # already list of strings
        entered = current_contacts - self.prev_contacts

        if entered:
            # log each new contact as an event
            for c in sorted(entered):
                obj = self._extract_object_name_from_contact(c)
                self.total_collision_events += 1
                self.total_unique_collisions.add(obj)
                self.ef.write(f"[t={time.time()-self.t0:.3f}s step={step}] COLLISION_ENTER {obj} via {c}\n")

        self.prev_contacts = current_contacts
        nearby = []
        for o in objects:
            oname = getattr(o, "name", str(o))
            lname = oname.lower()
            if any(s in lname for s in self.ignore_name_substrings):
                continue
            d, _ = self._approx_surface_distance(robot_pos, o)
            if d is None:
                continue
            if d <= self.safety_radius:
                nearby.append((float(d), oname))

        nearby.sort(key=lambda x: x[0])
        nearby_top = nearby[:self.near_limit]

        # Write a periodic “nearby” line (not every step unless you want)
        # if step % max(1, self.log_every * 10) == 0:
        #     if nearby_top:
        #         items = ", ".join([f"{name}({dist:.2f}m)" for dist, name in nearby_top])
        #         self.ef.write(f"[t={time.time()-self.t0:.3f}s step={step}] NEARBY {items}\n")

        nearby_signature = tuple((name, round(dist, 2)) for dist, name in nearby_top)

        if nearby_top:
            items = ", ".join([f"{name}({dist:.2f}m)" for dist, name in nearby_top])
            line = f"[t={time.time()-self.t0:.3f}s step={step}] NEARBY {items}\n"

            # Only write if changed from last time
            if nearby_signature != self.last_nearby_signature:
                self.ef.write(line)
                self.ef.flush()
                self.last_nearby_signature = nearby_signature
                self.last_nearby_line = line



        row = {
            "type": "step",
            "t_wall": time.time() - self.t0,
            "step": int(step),
            "sim_time": getattr(env, "sim_time", None),
            "robot_pos": self._to_list(robot_pos),
            "robot_orn": self._to_list(robot_orn),
            "robot_states": robot_states,
            "min_dist": min_d,
            "min_center_dist": min_center,
            "closest_object": closest,
            "safety_radius": self.safety_radius,
            "safety_violation": safety_violation,
            "flagged_objects": flagged_objects,
        }
        contacts = robot_states.get("ContactBodies", [])

        # Normalize to list for stable logging
        # if isinstance(contacts, set):
        #     contacts = sorted(list(contacts))
        # elif not isinstance(contacts, list):
        #     contacts = [contacts]

        row["num_contacts"] = len(contacts)
        row["contact_bodies_top5"] = contacts[:5]
        row["nearby_top"] = [{"name": n, "dist": d} for d, n in nearby_top]
        row["total_collision_events"] = self.total_collision_events
        row["unique_collisions_count"] = len(self.total_unique_collisions)


        self._write(row)


    #dump labels
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

            # sample of attributes on the states object
            try:
                info["states_dir_sample"] = [x for x in dir(st) if "key" in x.lower() or "state" in x.lower()][:30]
            except Exception:
                pass

            # try dict-like
            if hasattr(st, "keys"):
                try:
                    info["keys"] = sorted(list(st.keys()))
                    return info
                except Exception:
                    pass

            # try internal dict-ish attributes
            for attr in ("state_dict", "_states", "_state_dict", "states"):
                if hasattr(st, attr):
                    try:
                        d = getattr(st, attr)
                        if hasattr(d, "keys"):
                            info["keys"] = sorted(list(d.keys()))
                            return info
                    except Exception:
                        pass

            # try iterable of state objects
            if hasattr(st, "__iter__"):
                try:
                    names = []
                    for s in st:
                        names.append(getattr(s, "name", str(s)))
                    info["keys"] = sorted(names)
                    return info
                except Exception:
                    pass

            return info

        obj_list = list(objects)
        obj_names = [getattr(o, "name", str(o)) for o in obj_list[:max_obj]]

        robot_info = describe_states(robot)

        obj_infos = {}
        for o in obj_list[:max_obj]:
            obj_infos[getattr(o, "name", str(o))] = describe_states(o)

        self._write({
            "type": "available_states",
            "t_wall": time.time() - self.t0,
            "num_objects_passed": len(obj_list),
            "first_object_names": obj_names,
            "robot_states_info": robot_info,
            "object_states_info": obj_infos,
        })
