# Architecture

[日本語](architecture.ja.md)

```text
YAML / CLI
    ↓
Orchestrator (orchestrator.py | async_rl/orchestrator.py)
    ├── VerifiersRuntime      rollout/runtime.py   (Environment, serving(), episode(task, ctx, n))
    │     └── PolicyClient    rollout/client.py    (vf.Client: renderer -> PolicyBackend -> Response)
    ├── PolicyBackend         policy/base.py       (token ids in, token ids + logprobs out)
    ├── RolloutGroup          rollout/schema.py    (canonical; no torch/verifiers imports)
    ├── RunLogger             logging/             (run directory, jsonl, summary, lineage)
    └── TrainerBackend        trainer/base.py
          └── unsloth-native  trainer/unsloth_native/   TRL GRPOTrainer (+Unsloth patches) shim
```

The orchestrator owns sequencing, logging and checkpoints only. Verifiers owns rollouts; the
trainer backend owns the model, loss and optimizer; the policy backend owns generation.
`VerifiersRuntime` is the only piece that touches `verifiers`/`renderers`; trainer backends
never import it, and `rollout/schema.py` never imports torch, verifiers, or unsloth — it is the
seam between the two halves of the project. Backend calls (`setup`, `train_step`,
`save_checkpoint`, `load_checkpoint`) run through `asyncio.to_thread`, so a backend with its
own event loop does not conflict with the orchestrator's.

## Canonical rollout schema

Defined in `src/moongazer_rl/rollout/schema.py`.

| type | what it is |
|---|---|
| `ModelCall` | One policy call inside a rollout, exactly as sampled: `prompt_ids`, `completion_ids`, `sampling_logprobs` (aligned 1:1 with `completion_ids`), `finish_reason`, `policy_version`, timing. |
| `TrainingSequence` | One lossless token path (a verifiers `Branch`): `token_ids` + a per-token `action_mask` (`False` for tool results / user-simulator turns / template scaffold, `True` for sampled tokens) + `sampling_logprobs` aligned to `token_ids`. A trajectory has one sequence when every turn's prompt was an exact token prefix of the previous turn; otherwise verifiers forks the graph and there are several. |
| `Trajectory` | One verifiers rollout (one `Trace`): reward, `reward_components` (verbatim, never re-summed), metrics, info, errors, turn/token counts, timing, plus both `model_calls[]` and `sequences[]`. `is_trainable` = complete logprobs and a finite reward. A trace whose scoring raised gets `reward=NaN` and `metadata.scoring_failed=true`: logged, never trained on as a 0.0. |
| `RolloutGroup` | **`1 Task x G rollouts = 1 RolloutGroup`** — the GRPO unit. Holds `trajectories` (indices `0..G-1`), `task_id` (`"<taskset id>:<task idx>"`), `policy_version` (weight version when the episode started), `reward_mean` / `reward_std` (population std, matching TRL's `scale_rewards="group"`). |

Token fidelity rule: nothing is ever re-tokenized. `ModelCall` carries exactly the token IDs
and logprobs the policy backend returned at sampling time; `TrainingSequence` carries the
verifiers graph's own token path. `rollout/convert.py::verify_fidelity` cross-checks the two
views per trajectory and records any mismatch in `trajectory.metadata["fidelity_problems"]`
for the `rollouts.jsonl` audit. Before row construction,
`rollout/convert.py::verify_training_fidelity` gates the view selected by `trainer.call_unit`:
call mode requires finite, aligned logprobs on every non-empty `ModelCall`; sequence mode
requires aligned sequence arrays, finite logprobs at action positions, and ordered token/logprob
agreement with the calls that cover every action span. Both modes require one policy version
within the trajectory and agreement with the group. A failing trajectory is excluded before the
group baseline is computed; the remaining trajectories train unless fewer than the estimator's
minimum group size remain. Drops are counted, returned in `StepResult.extra`, and warned once per
step.
