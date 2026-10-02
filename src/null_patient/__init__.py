"""Top-level alias so verifiers can resolve ``harness.id: null_patient``.

verifiers' plugin loader probes ``verifiers.v1.harnesses.<id>`` with ``importlib.util.find_spec``,
which raises for a dotted id whose first segment is not a verifiers subpackage (for
``moongazer_rl.harnesses.null_patient`` it fails on
``verifiers.v1.harnesses.moongazer_rl``), so the plugin needs a flat module name. The
implementation lives in ``moongazer_rl.harnesses.null_patient``."""

from moongazer_rl.harnesses.null_patient import NullPatientHarness, NullPatientHarnessConfig

__all__ = ["NullPatientHarness", "NullPatientHarnessConfig"]
