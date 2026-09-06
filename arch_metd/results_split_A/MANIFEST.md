# Run A — kept draw

Snapshot of `arch_metd/results/` taken 2026-09-06, before the splitter was replaced.

| | |
|---|---|
| array fingerprint | `3d017f08c581c5fd` |
| **split hash (this machine)** | `22810e2548e07f90` |
| local scikit-learn | 1.9.0 |
| splitter | sklearn `GroupKFold` / `StratifiedGroupKFold` |

**Kept deliberately, not superseded.** Run A and the coming Run B are two arbitrary but
valid draws from the space of subject assignments. A result that survives both is real; a
result that flips between them was never a result. That is a robustness check obtained for
free, not a reason to discard either draw.

The workstation's own split for this run differed from this machine's on 17 of 169
subjects (all SAM40/CogLoad) because sklearn's `GroupKFold` breaks group-size ties with an
unstable argsort in some versions — SAM40's subjects all have exactly 12 trials. Pinning
alone would version-lock that tie-break and call a coin flip reproducible, so the splitter
is being replaced with a deterministic greedy assignment for Run B.

Headline figures from this draw are in `../REPORT.md` at the time of snapshot.
