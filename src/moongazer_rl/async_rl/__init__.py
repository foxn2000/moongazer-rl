"""Fully asynchronous RL: rollout producers and the learner run as independent loops joined by
a bounded queue. Separate execution path from ``orchestrator.py`` (the synchronous loop), which
is left untouched."""
