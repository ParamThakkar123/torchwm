"""Snapshot and restore utilities for the simulator.

This module centralizes snapshot format helpers so adapters and envs can
produce interoperable snapshots containing physics state and RNGStreams
state.
"""

from __future__ import annotations

from typing import Any, Dict


def make_snapshot(
    physics_state: Dict[str, Any], rng_states: Dict[str, Any], version: str = "1.0"
) -> Dict[str, Any]:
    return {"version": version, "physics": physics_state, "rngs": rng_states}


def parse_snapshot(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    # Basic validation/compatibility checks can go here
    if not isinstance(snapshot, dict):
        raise ValueError("snapshot must be a dict")
    return snapshot
