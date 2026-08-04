"""Small, testable contracts for gradient-accumulated training."""

from __future__ import annotations


def accumulation_divisor(batch_index: int, num_batches: int, accumulation_steps: int) -> int:
    """Scale the final partial accumulation window without underweighting it."""
    if accumulation_steps < 1 or not 0 <= batch_index < num_batches:
        raise ValueError("invalid accumulation schedule")
    remainder = num_batches % accumulation_steps
    if remainder and batch_index >= num_batches - remainder:
        return remainder
    return accumulation_steps


def optimizer_step_due(batch_index: int, num_batches: int, accumulation_steps: int) -> bool:
    """Return whether this microbatch closes an accumulation window."""
    accumulation_divisor(batch_index, num_batches, accumulation_steps)
    return (batch_index + 1) % accumulation_steps == 0 or batch_index + 1 == num_batches


def epoch_requires_training(epoch: int, pending_validation_epoch: int | None) -> bool:
    """Skip repeated training when resuming a checkpoint awaiting validation."""
    return pending_validation_epoch != epoch + 1
