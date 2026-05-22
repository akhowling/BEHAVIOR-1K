import copy
import numpy as np


class AttackInjector:
    def __init__(
        self,
        mode="none",
        seed=0,

        # No. 1 action noise injection
        action_random_prob=0.0,
        action_noise_sigma=0.0,

        # No. 3 random image masks
        rgb_cutout_prob=0.0,
        rgb_cutout_num_boxes=0,
        rgb_cutout_box_frac=0.2,
        rgb_key="rgb",
    ):
        self.mode = mode
        self.rng = np.random.default_rng(seed)

        self.action_random_prob = action_random_prob
        self.action_noise_sigma = action_noise_sigma

        self.rgb_cutout_prob = rgb_cutout_prob
        self.rgb_cutout_num_boxes = rgb_cutout_num_boxes
        self.rgb_cutout_box_frac = rgb_cutout_box_frac
        self.rgb_key = rgb_key

        self.seed = seed

    def corrupt_obs(self, obs):
        """
        Corrupt observation before policy inference.
        This implements No. 3 random image masks.
        """

        attack_info = {
            "obs_attack_enabled": False,
            "obs_attack_type": None,
            "rgb_key": self.rgb_key,
            "rgb_cutout_applied": False,
            "rgb_cutout_num_boxes": self.rgb_cutout_num_boxes,
            "rgb_cutout_box_frac": self.rgb_cutout_box_frac,
        }

        if self.mode not in ["rgb_cutout", "combined"]:
            return obs, attack_info

        if self.rgb_cutout_prob <= 0.0 or self.rgb_cutout_num_boxes <= 0:
            return obs, attack_info

        if self.rng.random() > self.rgb_cutout_prob:
            return obs, attack_info

        obs2 = copy.deepcopy(obs)

        if self.rgb_key not in obs2:
            attack_info["obs_attack_enabled"] = True
            attack_info["obs_attack_type"] = "rgb_random_cutout"
            attack_info["rgb_cutout_applied"] = False
            attack_info["error"] = f"rgb_key {self.rgb_key} not found in obs"
            return obs2, attack_info

        img = obs2[self.rgb_key]

        if img is None or not hasattr(img, "shape") or len(img.shape) < 2:
            attack_info["obs_attack_enabled"] = True
            attack_info["obs_attack_type"] = "rgb_random_cutout"
            attack_info["rgb_cutout_applied"] = False
            attack_info["error"] = "rgb image invalid"
            return obs2, attack_info

        obs2[self.rgb_key] = self.random_cutout(img)

        attack_info["obs_attack_enabled"] = True
        attack_info["obs_attack_type"] = "rgb_random_cutout"
        attack_info["rgb_cutout_applied"] = True

        return obs2, attack_info

    def corrupt_action(self, action, action_space=None):
        import copy
        import numpy as np
        import torch

        original_action = copy.deepcopy(action)
        executed_action = copy.deepcopy(action)

        attack_info = {
            "enabled": self.mode in ["action_noise", "combined"],
            "attack_type": None,
            "action_random_prob": self.action_random_prob,
            "action_noise_sigma": self.action_noise_sigma,
            "random_replace_applied": False,
            "gaussian_noise_applied": False,
        }

        if self.mode not in ["action_noise", "combined"]:
            attack_info["executed_action_changed"] = False
            return executed_action, attack_info

        def corrupt_leaf(x):
            if isinstance(x, torch.Tensor):
                y = x.clone()

                if self.action_random_prob > 0.0 and self.rng.random() < self.action_random_prob:
                    rand = self.rng.uniform(-1.0, 1.0, size=tuple(y.shape))
                    y = torch.tensor(rand, dtype=y.dtype, device=y.device)
                    attack_info["random_replace_applied"] = True

                if self.action_noise_sigma > 0.0:
                    noise_np = self.rng.normal(0.0, self.action_noise_sigma, size=tuple(y.shape))
                    noise = torch.tensor(noise_np, dtype=y.dtype, device=y.device)
                    y = y + noise
                    attack_info["gaussian_noise_applied"] = True

                return y

            if isinstance(x, np.ndarray):
                y = x.copy()

                if self.action_random_prob > 0.0 and self.rng.random() < self.action_random_prob:
                    y = self.rng.uniform(-1.0, 1.0, size=y.shape).astype(y.dtype)
                    attack_info["random_replace_applied"] = True

                if self.action_noise_sigma > 0.0:
                    y = y + self.rng.normal(0.0, self.action_noise_sigma, size=y.shape)
                    attack_info["gaussian_noise_applied"] = True

                return y

            if isinstance(x, float) or isinstance(x, int):
                y = float(x)

                if self.action_random_prob > 0.0 and self.rng.random() < self.action_random_prob:
                    y = float(self.rng.uniform(-1.0, 1.0))
                    attack_info["random_replace_applied"] = True

                if self.action_noise_sigma > 0.0:
                    y = y + float(self.rng.normal(0.0, self.action_noise_sigma))
                    attack_info["gaussian_noise_applied"] = True

                return y

            return x

        def recurse(x):
            if isinstance(x, dict):
                return {k: recurse(v) for k, v in x.items()}
            if isinstance(x, list):
                return [recurse(v) for v in x]
            if isinstance(x, tuple):
                return tuple(recurse(v) for v in x)
            return corrupt_leaf(x)

        executed_action = recurse(executed_action)

        types = []
        if attack_info["random_replace_applied"]:
            types.append("action_random_replace")
        if attack_info["gaussian_noise_applied"]:
            types.append("action_gaussian_noise")

        attack_info["attack_type"] = "+".join(types) if types else None
        attack_info["executed_action_changed"] = bool(types)

        return executed_action, attack_info
    def random_cutout(self, img):
        import torch
        import numpy as np

        is_torch = isinstance(img, torch.Tensor)

        if is_torch:
            out = img.clone()
        else:
            out = img.copy()

        h, w = out.shape[:2]

        box_h = max(1, int(self.rgb_cutout_box_frac * h))
        box_w = max(1, int(self.rgb_cutout_box_frac * w))

        for _ in range(self.rgb_cutout_num_boxes):
            y0 = int(self.rng.integers(0, max(1, h - box_h + 1)))
            x0 = int(self.rng.integers(0, max(1, w - box_w + 1)))

            y1 = min(h, y0 + box_h)
            x1 = min(w, x0 + box_w)

            out[y0:y1, x0:x1] = 0

        return out