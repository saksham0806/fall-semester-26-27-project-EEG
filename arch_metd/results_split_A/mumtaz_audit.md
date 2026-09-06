# Mumtaz acquisition audit — §6.5 item 3

Measured 2026-09-06 from the EDF headers of all 120 EC/EO recordings (63 subjects).

## The class is confounded with the recording setup

| header field | separates H from MDD | note |
|---|---:|---|
| **number of signals (21 vs 23)** | **0.8889** | 23 sig: 24 H / 1 MDD · 21 sig: 6 H / 32 MDD |
| recording year (2011 vs 2013) | 0.6825 | all ten 2011 recordings are H, none MDD |
| equipment serial (`SN:060224`) | 0.5238 | constant — same machine throughout |
| majority-class baseline | 0.5238 | |

Healthy controls were overwhelmingly recorded on a 23-signal montage and patients on a
21-signal one. The six frontal channels are selected **by name** and resolve in every
file, so the extra channels are not read — but a different montage means a different
acquisition configuration, and it is 89% collinear with the diagnosis.

## What that does to the 0.9418

| | AUC | 95% CI | n |
|---|---:|---|---|
| all Mumtaz subjects | 0.9418 | [0.874, 0.991] | 61 (32 MDD / 29 HC) |
| **within 21-signal montage only** | 0.9107 | [0.783, 1.000] | 32 (28 MDD / **4 HC**) |
| **within 23-signal montage only** | 0.8472 | [0.462, 1.000] | 27 (**3 MDD** / 24 HC) |
| montage alone, no EEG at all | 0.8685 | — | 61 |

**Read this carefully in both directions.** Performance does not collapse inside a single
montage, so the model is not *only* reading the setup. But montage alone scores 0.8685
against the model's 0.9418, and neither within-montage estimate is powered to settle the
residual: one has 4 healthy subjects, the other 3 patients, and the 23-signal CI crosses
chance. **Mumtaz cannot deconfound montage from diagnosis, because no adequately sized
subgroup contains both classes on one montage.**

Consistent with the §8.3 controls: a never-trained random backbone reaches 0.9235 there,
and recording length alone predicts class at 0.709 (MODMA: 0.514).

## Consequence

Report Mumtaz's depression AUC only alongside this table. The honest claim is that a
substantial part of it is acquisition, that the remainder cannot be measured on this
corpus, and that the MODMA↔Mumtaz transfer in P7 inherits the problem — a model trained
on MODMA and tested on Mumtaz is tested against a target whose classes differ in setup.
