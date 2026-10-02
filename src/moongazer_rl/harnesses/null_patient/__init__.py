"""verifiers harness plugin: the stock ``null`` chat loop with a configurable per-call timeout
and no SDK retries (see harness.py). Resolved by ``harness.id: null_patient`` through the
top-level alias package ``src/null_patient``."""

from moongazer_rl.harnesses.null_patient.harness import (
    NullPatientHarness,
    NullPatientHarnessConfig,
)

__all__ = ["NullPatientHarness", "NullPatientHarnessConfig"]
