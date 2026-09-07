# ADR-002: Two NaN policies, because the two consumers mean different things

**Status:** accepted
**Date:** 2026-09-05

## Context

The released CSVs contain non-finite values: 0.24% of rows have NaN in the
seven dispersion columns (0.83% at 5 nodes, 0.00% at 20), and 560 rows have
NaN in a mean column. The dispersion pattern is consistent with a sample
standard deviation over fewer than two reporting nodes.

The upstream pipeline replaces every NaN with 0.0, in `safe_minmax_normalize`
and again via `torch.nan_to_num`.

## Decision

**Model input:** follow the release's finite-input convention and fill with 0.0.
The broader primary pipeline is corrected rather than claimed as bit-for-bit
upstream reproduction.

**Rule scoring:** NaN-aware. A missing dispersion means dispersion is undefined,
not zero. Rules use available observations; an entirely unobserved signal makes
no positive assertion and therefore returns neutral zero.

## Consequences

The raw/normalized split from ADR-001 makes the policies separable. Neutral zero
means “no rule evidence,” never “protocol compliant” or “benign.”

Harder: the two paths must stay in step. A test asserts that model input is
finite everywhere and that rule scores are finite everywhere, on every
domain, so a regression in either shows up immediately.

Given up: a single uniform policy, which would have been simpler to explain
but would have forced one of the two consumers to accept a wrong answer.

## Alternatives considered

*Drop windows containing NaN.* Rejected: the loss is small overall but falls
disproportionately on 5-node domains, which correlates with domain identity and
would bias the continual comparison in a way that is hard to bound.

*Impute dispersion from neighboring minutes.* Rejected: it invents observations,
and R3 could then fire on the imputation rather than the network.
