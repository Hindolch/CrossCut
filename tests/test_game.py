import base64
import io
import unittest

from PIL import Image

from crosscut.game import CrafterGame


class CrafterGameTests(unittest.TestCase):
    def test_initial_state_requires_no_action_and_exposes_only_visible_window(self):
        game = CrafterGame(seed=123)
        state = game.snapshot()
        self.assertEqual(state["steps"], 0)
        self.assertEqual(game.env._step, 0)
        self.assertEqual(state["player_pos"], [32, 32])
        self.assertEqual(state["facing"], [0, 1])
        self.assertEqual(state["inventory"]["health"], 9)
        self.assertFalse(any(state["achievements"].values()))
        self.assertEqual(len(state["local_map"]), 7)
        self.assertTrue(all(len(row) == 9 for row in state["local_map"]))
        self.assertEqual(state["local_map"][3][4], "player-down")
        self.assertNotIn("semantic", state)
        with Image.open(io.BytesIO(base64.b64decode(state["image_png_base64"]))) as image:
            self.assertEqual(image.size, (256, 256))
        self.assertNotIn("image_png_base64", game.snapshot(False))

    def test_step_validation_and_episode_boundary(self):
        game = CrafterGame(seed=123, length=1)
        for action in ("teleport", -1, None, []):
            with self.assertRaises(ValueError):
                game.step(action)
        self.assertEqual(game.steps, 0)
        result = game.step("move_left")
        self.assertEqual(result["steps"], 1)
        self.assertEqual(game.env._step, 1)
        self.assertEqual(result["facing"], [-1, 0])
        self.assertTrue(result["done"])
        with self.assertRaises(RuntimeError):
            game.step("noop")
        self.assertEqual(game.steps, 1)

    def test_polling_preserves_rng_and_same_seed_replay(self):
        first = CrafterGame(seed=654)
        replay = CrafterGame(seed=654)
        self.assertEqual(first.snapshot(), replay.snapshot())
        # Includes nighttime rendering. Polling must never invoke env.render(),
        # which would consume world RNG through its nighttime noise effect.
        actions = ["move_left", "do", "move_right", "do", "sleep", "noop"] * 50
        for action in actions:
            if first.done or first.success:
                break
            before = first.env._world.random.get_state()
            for _ in range(3):
                first.snapshot()
                first.snapshot(False)
            after = first.env._world.random.get_state()
            self.assertEqual(before[0], after[0])
            self.assertTrue((before[1] == after[1]).all())
            self.assertEqual(before[2:], after[2:])
            self.assertEqual(first.step(action), replay.step(action))


if __name__ == "__main__":
    unittest.main()
