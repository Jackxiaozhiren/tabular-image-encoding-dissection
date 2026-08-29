# DMKD v1.0.0 release checklist

- [x] Canonical manuscript package identified as the 2026-08-20 `DMKD_ready_to_submit` state.
- [x] Public repository mapping corrected to `Jackxiaozhiren/tabular-image-encoding-dissection`.
- [x] Manuscript-active 15-dataset code synchronized from the audited research bundle.
- [x] Adult M3-full 0.9091 vs M3-noG 0.8471 reconciled against frozen evidence.
- [x] Bank M3-full 0.9249 vs M3-noG 0.8665 reconciled against frozen evidence.
- [x] Categorical-channel Wilcoxon p=3.8016e-05 and dataset-median p=0.00390625 reconciled.
- [x] Same-feature MLP vs M3-full p=0.1256 reconciled.
- [x] ResNet dataset-median sensitivity p=0.064453125 reconciled.
- [x] `08_extended_evaluate.py`, `11_resnet_analysis.py`, and `make_tables.py` executed successfully against the frozen evidence.
- [x] Encoder tests and frozen-result verification pass.
- [x] Raw datasets, row-level predictions, processed NPZ files, checkpoints, caches, and submission-management artifacts excluded.
- [x] `CITATION.cff` frozen as version 1.0.0 with release date 2026-08-29.
- [ ] Commit deterministic `RELEASE_MANIFEST.sha256` generated from the exact final candidate tree.
- [ ] Require CI success with exact manifest equality on the final candidate commit.
- [ ] Merge PR #1 to `main`.
- [ ] Publish GitHub release/tag `v1.0.0` on the merged commit.
- [ ] Update the repository About description from the historical Neurocomputing wording to the DMKD study wording.

## Manuscript wording note

A manuscript sentence that says the scalability section reports training/inference time and throughput is broader than the evidence currently frozen here; the audited scalability material reports analytical complexity/parameter information and explicitly states that no new timing benchmark is introduced. If the manuscript remains editable, reconcile that sentence without altering the experimental results.
