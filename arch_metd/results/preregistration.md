# Pre-registration — manifold gate (§8.2)

Recorded **2026-09-06, before the corpus silhouette was measured.** Nothing in this file
may be revised after the measurement; a criterion chosen once the number is known is not a
criterion.

## Criterion

The shared backbone is retained only if, at lambda = 1:

    silhouette(anchor embeddings, labelled by corpus)  <  silhouette(randbb anchors) - 0.05

The absolute value is reported alongside, against §8.2's proposed **0.3**, so the result
stays comparable to the specification. The **decision** rests on the relative test.

## Why relative rather than absolute

A frozen random backbone sets the floor for how corpus-separable this input already is
before any training. If the adversarially-trained backbone does not sit below that floor,
adaptation achieved nothing, whatever the absolute silhouette happens to be. This follows
§12 item 9 — compare against your own control, never against an assumed number — and the
control is already trained (`model_fold*_randbb.pt`).

## Consequence if it fails

Fall back to independent single-task backbones, per §8.2. Note this is already the
direction the §8.3 controls point: the random backbone matched or beat the trained one on
5 of 7 aggregate metrics, and 33,287 of the model's 34,487 parameters live in the heads
rather than the 1,200-parameter trunk.

## Reported regardless of verdict

The delta between lambda = 0 and lambda = 1, not a single verdict.

---

## Outcome — measured 2026-09-06, after the above was fixed

| mode | anchor probe (chance 0.250) | full-set probe (majority 0.4943) | silhouette |
|---|---:|---:|---:|
| main (λ=1) | 0.3970 | 0.5626 | −0.0028 |
| lambda0 (λ=0) | 0.4761 | 0.6053 | +0.0025 |
| randbb (random) | 0.4649 | 0.5997 | +0.0033 |

**Gate: FAIL.** main −0.0028 is not below randbb − 0.05 = −0.0467.

**Honest caveat on the criterion, stated without revising it.** Every silhouette lands
within ±0.004 of zero, so the 0.05 margin was far larger than the entire observed range
and the test could not have passed under any outcome short of dramatic anti-clustering.
The silhouette is uninformative on this data — corpus structure is present but diffuse
rather than compactly clustered, which is why §8.2's absolute 0.3 threshold would have
passed trivially in the same breath. That is a property of the metric here, not evidence
that adaptation worked. The criterion stands as recorded and the gate is failed; the
probes below are what carry the actual finding.

---

# Pre-registration 2 — replacing §8.2 (§6.5 item 4)

Recorded **2026-09-06**, before the fixed pipeline (canonical split + class-weighted
heads) has been re-run. The numbers quoted below come from the superseded run and are
stated only to show the criterion is achievable and discriminating; they are **not** the
measurement this gate applies to.

## Why §8.2 is replaced rather than repeated

The silhouette gate could not adjudicate anything. All three modes landed within ±0.004
of zero, so the pre-registered relative test could not pass and §8.2's absolute 0.3 would
have passed trivially. Corpus structure in this representation is diffuse rather than
compactly clustered, which the silhouette is the wrong instrument to see. The frozen
probe separated the same three modes cleanly.

## Criterion

The shared backbone is retained only if, on the re-run:

    anchor_probe(main)  <  anchor_probe(lambda0) - 0.05

Both measured on the same folds, same architecture, backbone frozen, with an independent
logistic regression — never the adversary's own head (§12 item 9).

Superseded-run values, for calibration only: main 0.3970, λ=0 0.4761, margin 0.079,
balanced chance 0.250.

## What this criterion does and does not claim

It tests whether adversarial training removed domain information **relative to its own
method-off control**. It deliberately does not require the probe to reach the 0.250
chance floor: partial removal is a real effect and the protocol-gated inference in §0
stands regardless. A separate honest reporting obligation remains — the absolute anchor
and full-set probe values are reported alongside the verdict, so a passing margin on a
still-corpus-separable representation cannot be quoted as invariance.

## Consequence if it fails

Fall back to independent single-task backbones, per §8.2's original consequence.

## Reported regardless of verdict

Anchor probe and full-set probe at λ=1, λ=0 and random-backbone, with the margin.

## Outcome — measured 2026-09-06 on the fixed pipeline

| mode | anchor probe (chance 0.250) | full-set probe (majority 0.4943) |
|---|---:|---:|
| main (λ=1) | **0.3605** | 0.5388 |
| lambda0 (λ=0) | 0.4115 | 0.5776 |
| randbb (random) | 0.4649 | 0.5997 |

**Gate: PASS.** 0.3605 < 0.4115 − 0.05 = 0.3615, by a margin of **0.0010**.

**Robustness, reported because the criterion does not require it and a reader needs it.**
Only **2 of 5 folds** clear the threshold individually, and the per-fold margin is
0.0510 ± 0.0236 — the 0.05 bar sits essentially on the mean. The criterion is met as
written; the pass is not robust, and a re-run could plausibly return the other verdict.

**The reporting obligation, discharged.** The representation is *not* invariant: the
anchor probe remains **+0.1105 above balanced chance** and the full-set probe **+0.0445
above the majority domain**. Adversarial training removed some domain information relative
to its own control. It did not achieve invariance, and this gate never claimed it did.

**Read this next to the transfer result, which is the point of the exercise.** On the one
direction where the adversary is operative (SAM40+CogLoad → EEGMAT, §7.3 paired sign
test), λ=1 scores **0.4167** and λ=0 scores **0.6667**. Domain removal was achieved,
marginally — and it *cost* generalisation rather than buying it. The gate passing and the
architecture failing are not in conflict: the gate tested whether the mechanism does what
it says, and it does, weakly. Whether that mechanism helps is what the transfer test
answers, and the answer is no.

Superseded silhouette values, for the record: main −0.0001, λ=0 +0.0025, random +0.0033.
