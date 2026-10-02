"""Deterministic first-fit packing of training rows; no tensor dependencies."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass

from .dataset import TrainRow


@dataclass(frozen=True)
class BatchBudget:
    max_model_tokens: int | None = None
    max_action_tokens: int | None = None
    max_rows: int | None = None
    max_groups: int | None = None
    keep_groups_whole: bool = True
    length_bucketing: bool = True

    def __post_init__(self):
        for name in ("max_model_tokens", "max_action_tokens", "max_rows", "max_groups"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer or None")


def describe_plan(rows: Sequence[TrainRow], plan: list[list[int]]) -> list[dict]:
    return [
        dict(
            rows=len(chunk),
            model_tokens=sum(rows[i].total_len for i in chunk),
            action_tokens=sum(rows[i].num_action_tokens for i in chunk),
            groups=len({rows[i].group_index for i in chunk}),
        )
        for chunk in plan
    ]


def plan_microbatches(rows: Sequence[TrainRow], budget: BatchBudget) -> list[list[int]]:
    """Pack indivisible groups (or rows), retaining contiguous rows of each group."""
    grouped = {}
    for i, row in enumerate(rows):
        grouped.setdefault(row.group_index if budget.keep_groups_whole else i, []).append(i)
    units = list(grouped.values())
    if budget.length_bucketing:
        units.sort(key=lambda unit: -sum(rows[i].total_len for i in unit))

    def fits(chunk):
        counts = describe_plan(rows, [chunk])[0]
        # A micro-batch is forwarded at rows x its widest row (the trainer pads the rows of a
        # micro-batch to a common width), so that padded volume is what the token budget caps.
        counts["model_tokens"] = len(chunk) * max(rows[i].total_len for i in chunk)
        return all(
            getattr(budget, "max_" + name) is None or count <= getattr(budget, "max_" + name)
            for name, count in counts.items()
        )

    plan = []
    oversized = set()
    warned = False
    for unit in units:
        if not fits(unit):
            # The budget is a memory limit: a group that does not fit on its own is split
            # into consecutive budget-sized pieces (rows stay in group order). Advantages
            # are already per row, so the loss is unchanged; only PrefixGrouper's shared
            # prompt reuse is lost for that group.
            if not warned:
                warnings.warn(
                    "a RolloutGroup exceeds the micro-batch budget on its own; splitting it "
                    "across micro-batches (loss unchanged, prompt sharing lost for that group)",
                    UserWarning,
                    stacklevel=2,
                )
                warned = True
            piece: list[int] = []
            for i in unit:
                if piece and not fits(piece + [i]):
                    oversized.add(len(plan))
                    plan.append(piece)
                    piece = []
                piece.append(i)
            oversized.add(len(plan))
            plan.append(piece)
            continue
        for j, chunk in enumerate(plan):
            if j not in oversized and fits(chunk + unit):
                chunk.extend(unit)
                break
        else:
            plan.append(unit.copy())
    for chunk in plan:
        # Stable group order, including when row units were packed in length order.
        order = {rows[i].group_index: None for i in sorted(chunk)}
        rank = {group: j for j, group in enumerate(order)}
        chunk.sort(key=lambda i: (rank[rows[i].group_index], i))
    return plan
