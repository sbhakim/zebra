# ADR-001: Evaluate protocol-guided signals on raw features

**Status:** accepted
**Date:** 2026-09-05

## Context

The detector consumes per-feature min--max normalized windows. The three rule
signals are functions of message counts and their temporal or cross-node
variation. They are heuristics informed by RPL behavior, not formal invariants.

Per-feature min--max scaling divides `diss` and `disr` by *different* spans.
A relationship written on normalized features instead becomes a statement
about independently rescaled quantities and is difficult to interpret.

## Decision

Rule signals are computed from the **raw** windows, before normalization, and
stored on each split alongside the normalized
tensor the model consumes.

## Consequences

The scores retain their interpretation in message-count units and are computed
once at load rather than every epoch. Thresholds and scales are learned from
benign training windows; those six values are explicit 48-byte static metadata.
No gradient path enters either raw signal computation or calibration.

Harder: `Domain` carries a third tensor, and any change to windowing must keep
`x`, `y`, and `nu` index-aligned. A test pins that alignment.

Given up: end-to-end learning of the signal definitions. That would be a valid
different method, but it would add auxiliary state and change the interpretability
claim being tested here.

## Alternatives considered

*Compute on normalized features.* Rejected: per-feature scaling can reverse the
meaning of a count ratio and make the rule dependent on a domain's extrema.

*Share one normalization scale across message pairs.* Rejected: this couples
unrelated model inputs and silently changes the detector's input distribution.
