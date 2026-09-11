"""Read-only observation adapter for the unmodified Crafter 1.8.3 game.

Private player/world reads provide initial state without advancing time. Only
the same 9 by 7 terrain window as the default visual observation is exposed.
"""

import base64
import io

import crafter
from PIL import Image


class CrafterGame:
    def __init__(self, seed: int, length: int = 10000):
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        if isinstance(length, bool) or not isinstance(length, int) or length < 1:
            raise ValueError("length must be a positive integer")
        self.seed = seed
        self.env = crafter.Env(seed=seed, size=(256, 256), length=length)
        self.action_names = list(self.env.action_names)
        self.steps = 0
        self.done = False
        self.reward = 0.0
        self.total_reward = 0.0
        self._obs = self.env.reset()
        self._image_png_base64 = None

    @property
    def success(self):
        return self.env._player.achievements.get("collect_diamond", 0) > 0

    def step(self, action: str):
        if not isinstance(action, str) or action not in self.action_names:
            raise ValueError("unknown Crafter action")
        if self.done or self.success:
            raise RuntimeError("episode has ended or the diamond objective is complete")
        self._obs, reward, done, _ = self.env.step(self.action_names.index(action))
        self.steps += 1
        self.done = bool(done)
        self.reward = float(reward)
        self.total_reward += self.reward
        self._image_png_base64 = None
        return self.snapshot()

    def _local_map(self):
        player = self.env._player
        px, py = map(int, player.pos)
        rows = []
        for y in range(py - 3, py + 4):
            row = []
            for x in range(px - 4, px + 5):
                material, obj = self.env._world[(x, y)]
                row.append(str(obj.texture) if obj is not None else (material or "unknown"))
            rows.append(row)
        return rows

    def snapshot(self, include_image=True):
        player = self.env._player
        state = {
            "seed": self.seed,
            "steps": self.steps,
            "done": self.done,
            "success": bool(self.success),
            "inventory": {k: int(v) for k, v in player.inventory.items()},
            "achievements": {k: int(v) for k, v in player.achievements.items()},
            "reward": self.reward,
            "total_reward": self.total_reward,
            "player_pos": [int(v) for v in player.pos],
            "facing": [int(v) for v in player.facing],
            "local_map": self._local_map(),
            "action_names": list(self.action_names),
        }
        if include_image:
            if self._image_png_base64 is None:
                buffer = io.BytesIO()
                Image.fromarray(self._obs).save(buffer, format="PNG")
                self._image_png_base64 = base64.b64encode(buffer.getvalue()).decode("ascii")
            state["image_png_base64"] = self._image_png_base64
        return state
