from __future__ import annotations

import unittest

from scripts.probe_stickpull_three_phase_bc import ThreePhaseBCGate


def make_gate() -> ThreePhaseBCGate:
    return ThreePhaseBCGate(
        initial_coefficient=10.0,
        fallback_coefficient=0.0,
        recovery_coefficient=100.0,
        fallback_after_evals=5,
        fallback_success_threshold=0.0,
        recovery_success_threshold=0.2,
        progress_window=5,
        return_progress_threshold=0.1,
    )


class ThreePhaseBCGateTests(unittest.TestCase):
    def test_falls_back_when_success_and_return_are_stalled(self) -> None:
        gate = make_gate()

        for average_return in (-40.0, -39.0, -41.0, -40.0, -39.0):
            decision = gate.update(success=0.0, average_return=average_return)

        self.assertEqual(decision.phase, "exploration_fallback")
        self.assertEqual(decision.coefficient, 0.0)
        self.assertIn("initial_transfer->exploration_fallback", decision.transition)

    def test_return_progress_keeps_initial_transfer_active(self) -> None:
        gate = make_gate()

        for average_return in (1.0, 2.0, 4.0, 8.0, 16.0):
            decision = gate.update(success=0.0, average_return=average_return)

        self.assertEqual(decision.phase, "initial_transfer")
        self.assertEqual(decision.coefficient, 10.0)
        self.assertEqual(decision.transition, "")

    def test_success_recovers_strong_bc_from_fallback(self) -> None:
        gate = make_gate()
        for average_return in (-40.0, -39.0, -41.0, -40.0, -39.0):
            gate.update(success=0.0, average_return=average_return)

        decision = gate.update(success=0.2, average_return=1000.0)

        self.assertEqual(decision.phase, "retention_recovery")
        self.assertEqual(decision.coefficient, 100.0)
        self.assertIn("exploration_fallback->retention_recovery", decision.transition)

    def test_early_success_skips_fallback(self) -> None:
        gate = make_gate()

        decision = gate.update(success=0.2, average_return=100.0)

        self.assertEqual(decision.phase, "retention_recovery")
        self.assertEqual(decision.coefficient, 100.0)
        self.assertIn("initial_transfer->retention_recovery", decision.transition)


if __name__ == "__main__":
    unittest.main()
