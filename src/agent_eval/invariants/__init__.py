"""Invariants: deterministic, fail-closed assertions over agent trajectories."""

from __future__ import annotations

from agent_eval.invariants.base import BaseInvariant, InvariantRegistry, InvariantViolation
from agent_eval.invariants.compliance import (
    NoPII,
    NumericGrounding,
    PCICompliant,
    assert_no_pii,
    assert_numeric_grounding,
    assert_pci_compliant,
    extract_monetary_figures,
    find_pci,
    find_pii,
    find_unparseable_money,
    luhn_valid,
    mask_pci,
    mask_pii,
    verhoeff_checksum_valid,
)
from agent_eval.invariants.topology import (
    NoContextBloat,
    NoRoutingLoops,
    NoSwallowedToolFailures,
    SLABounds,
    ValidSpanDAG,
    assert_no_context_bloat,
    assert_no_routing_loops,
    assert_no_swallowed_tool_failures,
    assert_sla_bounds,
    assert_valid_dag,
    find_cycle,
)


def default_registry(
    *,
    max_latency_ms: float | None = None,
    max_tokens: int | None = None,
    max_hops: int = 8,
) -> InvariantRegistry:
    """The financial-grade default set. SLA bounds are added when limits are given."""
    registry = InvariantRegistry(
        [
            assert_no_pii(),
            assert_pci_compliant(),
            assert_numeric_grounding(),
            assert_valid_dag(),
            assert_no_routing_loops(max_hops=max_hops),
            assert_no_context_bloat(),
            assert_no_swallowed_tool_failures(),
        ]
    )
    if max_latency_ms is not None or max_tokens is not None:
        registry.register(assert_sla_bounds(max_latency_ms=max_latency_ms, max_tokens=max_tokens))
    return registry


__all__ = [
    "BaseInvariant",
    "InvariantRegistry",
    "InvariantViolation",
    "NoContextBloat",
    "NoPII",
    "NoRoutingLoops",
    "NoSwallowedToolFailures",
    "NumericGrounding",
    "PCICompliant",
    "SLABounds",
    "ValidSpanDAG",
    "assert_no_context_bloat",
    "assert_no_pii",
    "assert_no_routing_loops",
    "assert_no_swallowed_tool_failures",
    "assert_numeric_grounding",
    "assert_pci_compliant",
    "assert_sla_bounds",
    "assert_valid_dag",
    "default_registry",
    "extract_monetary_figures",
    "find_cycle",
    "find_pci",
    "find_pii",
    "find_unparseable_money",
    "luhn_valid",
    "mask_pci",
    "mask_pii",
    "verhoeff_checksum_valid",
]
