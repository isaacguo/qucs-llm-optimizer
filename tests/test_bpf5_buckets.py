# tests/test_bpf5_buckets.py
from __future__ import annotations

import random
import unittest

from jobs.bpf5_agent.buckets import (
    BW_CHOICES_MHZ,
    BUCKET_WIDTH_MHZ,
    MAX_STEPS,
    N_BUCKETS,
    SUCCESS_PER_BUCKET,
    SWEEP_MHZ,
    pick_bucket,
    sample_goal,
)
from jobs.bpf5_agent.goals import goal_from_dict, validate_goal


class TestBpf5Buckets(unittest.TestCase):
    def test_constants(self):
        self.assertEqual(N_BUCKETS, 20)
        self.assertEqual(BUCKET_WIDTH_MHZ, 15)
        self.assertEqual(SWEEP_MHZ, (0.0, 300.0))
        self.assertEqual(BW_CHOICES_MHZ, (5, 10, 15))
        self.assertEqual(SUCCESS_PER_BUCKET, 10)
        self.assertEqual(MAX_STEPS, 19)

    def test_sample_passband_inside_sweep(self):
        rng = random.Random(0)
        for b in range(20):
            for _ in range(50):
                goal, cf, bw = sample_goal(b, rng)
                half = bw / 2
                self.assertGreaterEqual(cf - half, 0.0)
                self.assertLessEqual(cf + half, 300.0)
                self.assertEqual(goal["f_low_hz"], (cf - half) * 1e6)

    def test_sample_goal_cf_in_bucket_and_bw_choice(self):
        rng = random.Random(1)
        for b in range(N_BUCKETS):
            lo = b * BUCKET_WIDTH_MHZ
            hi = (b + 1) * BUCKET_WIDTH_MHZ
            for _ in range(30):
                goal, cf, bw = sample_goal(b, rng)
                self.assertIn(bw, BW_CHOICES_MHZ)
                self.assertGreaterEqual(cf, lo)
                self.assertLess(cf, hi) if b < N_BUCKETS - 1 else self.assertLessEqual(cf, hi)
                half = bw / 2.0
                self.assertEqual(goal["f_high_hz"], (cf + half) * 1e6)
                spec = goal_from_dict(goal)
                validate_goal(spec)

    def test_pick_bucket_prefers_underfilled(self):
        counts = [SUCCESS_PER_BUCKET] * N_BUCKETS
        counts[3] = 2
        counts[7] = 9
        rng = random.Random(42)
        seen = {pick_bucket(counts, rng) for _ in range(40)}
        self.assertTrue(seen.issubset({3, 7}))
        self.assertEqual(seen, {3, 7})

    def test_pick_bucket_when_all_full(self):
        counts = [SUCCESS_PER_BUCKET] * N_BUCKETS
        rng = random.Random(0)
        b = pick_bucket(counts, rng)
        self.assertGreaterEqual(b, 0)
        self.assertLess(b, N_BUCKETS)


if __name__ == "__main__":
    unittest.main()
