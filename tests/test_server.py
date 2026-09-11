import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor

from crosscut.server import Conflict, SessionStore


class FakeGame:
    action_names = ["noop", "diamond", "die"]

    def __init__(self, seed, length):
        self.seed, self.length = seed, length
        self.steps = 0
        self.done = self.success = False

    def step(self, action):
        if self.done or self.success:
            raise RuntimeError("ended")
        self.steps += 1
        self.success = action == "diamond"
        self.done = action == "die" or self.steps >= self.length

    def snapshot(self, include_image=True):
        return dict(seed=self.seed, steps=self.steps, done=self.done, success=self.success)


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = SessionStore(self.directory.name, FakeGame)
        self.store.mutate("player", "reset", dict(request_id="reset", expected_revision=0, seed=42))

    def action(self, actions, request_id="step", revision=1):
        return self.store.mutate("player", "step", dict(request_id=request_id, expected_revision=revision, actions=actions))

    def test_response_loss_retry_does_not_repeat_actions(self):
        first = self.action(["noop", "noop"])
        self.assertEqual(first, self.action(["noop", "noop"]))
        self.assertEqual(first["steps"], 2)

    def test_request_id_cannot_be_reused_for_different_actions(self):
        self.action(["noop"])
        with self.assertRaises(Conflict):
            self.action(["die"])

    def test_stale_revision_does_not_advance_game(self):
        self.action(["noop"])
        with self.assertRaises(Conflict):
            self.action(["noop"], "other")
        self.assertEqual(self.store.state("player")["steps"], 1)

    def test_concurrent_actions_commit_only_one_revision(self):
        def call(i):
            try:
                return self.action(["noop"], str(i))
            except Conflict:
                return None
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(call, range(2)))
        self.assertEqual(sum(x is not None for x in results), 1)
        self.assertEqual(self.store.state("player")["steps"], 1)

    def test_success_stops_batch_and_preserves_winning_episode(self):
        state = self.action(["diamond", "die", "noop"])
        self.assertEqual(state["steps"], 1)
        self.assertTrue(state["success"])
        with self.assertRaises(Conflict):
            self.store.mutate("player", "reset", dict(request_id="again", expected_revision=2, seed=43))

    def test_restart_replays_committed_actions(self):
        before = self.action(["noop", "noop"])
        after = SessionStore(self.directory.name, FakeGame).state("player")
        self.assertEqual(before, after)

    def test_invalid_batch_has_no_partial_side_effects(self):
        with self.assertRaises(ValueError):
            self.action(["noop", "bad"])
        self.assertEqual(self.store.state("player")["steps"], 0)

    def test_session_path_traversal_rejected(self):
        with self.assertRaises(ValueError):
            self.store.state("../escape")


    def budget_store(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        store = SessionStore(directory.name, FakeGame, step_budget=3)
        store.mutate("player", "reset", dict(request_id="reset", expected_revision=0, seed=42))
        return directory, store

    def test_lifetime_budget_truncates_batch_and_forbids_next_step(self):
        _, store = self.budget_store()
        request = dict(request_id="step", expected_revision=1, actions=["noop"] * 10)
        result = store.mutate("player", "step", request)
        self.assertEqual(result["steps"], 3)
        self.assertEqual(result["total_steps"], 3)
        self.assertEqual(result["step_budget"], 3)
        self.assertTrue(result["budget_exhausted"])
        self.assertEqual(result, store.mutate("player", "step", request))
        with self.assertRaises(Conflict):
            store.mutate("player", "step", dict(request_id="extra", expected_revision=2, actions=["noop"]))
        with self.assertRaises(Conflict):
            store.mutate("player", "reset", dict(request_id="reset2", expected_revision=2, seed=43))
        self.assertEqual(store.games["player"].steps, 3)
        self.assertEqual(store.state("player"), result)

    def test_reset_does_not_replenish_budget_and_restart_preserves_total(self):
        directory, store = self.budget_store()
        store.mutate("player", "step", dict(request_id="one", expected_revision=1, actions=["noop"] * 2))
        reset = store.mutate("player", "reset", dict(request_id="reset2", expected_revision=2, seed=43))
        self.assertEqual(reset["steps"], 0)
        self.assertEqual(reset["total_steps"], 2)
        self.assertFalse(reset["budget_exhausted"])
        restarted = SessionStore(directory.name, FakeGame, step_budget=3)
        self.assertEqual(reset, restarted.state("player"))
        last = restarted.mutate("player", "step", dict(request_id="last", expected_revision=3, actions=["noop"] * 3))
        self.assertEqual(last["steps"], 1)
        self.assertEqual(last["total_steps"], 3)
        self.assertTrue(last["budget_exhausted"])
        again = SessionStore(directory.name, FakeGame, step_budget=3)
        self.assertEqual(last, again.state("player"))
        with self.assertRaises(Conflict):
            again.mutate("player", "step", dict(request_id="extra", expected_revision=4, actions=["noop"]))
        self.assertEqual(again.games["player"].steps, 1)

    def test_restart_cannot_increase_persisted_budget(self):
        directory, _ = self.budget_store()
        with self.assertRaises(Conflict):
            SessionStore(directory.name, FakeGame, step_budget=4).state("player")

    def test_budget_cannot_exceed_hard_limit(self):
        for budget in (1000001, 0, True, 3.5):
            with self.assertRaises(ValueError):
                SessionStore(self.directory.name, FakeGame, step_budget=budget)


if __name__ == "__main__":
    unittest.main()
