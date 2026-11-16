# Archive gaps and provenance notes

| Item | Status |
|---|---|
| C32 production solver | The archive contains `solver/biot_mechanism_audit.py` (a full 16×16, 3-step backward-Euler Biot mechanism implementation) and `solver/c32_linked_storage_audit.py`. The original C32 production driver itself is **not** in this archive. |
| C73/C74 flow code | Present under `flow/`. |
| Frozen checkpoints | 57 `.pt` files under `checkpoints/`. |
| dev/validation/confirmation splits | Not stored as JSON. Split definitions are embedded in the C69/C74 dataset files and the archived evaluation scripts. |
| Cohort A/B/C/D | Present as `.npz` under `data/cohorts/`. |
| Result tables | Tables 5, 6, 8, 9, 10 reconstructed exactly; Table 7 is a tidy reconstruction of the learning-rate audit + Cohort C contrast block. |

## Before making the paper's availability statements point here

- Checkpoints total ~60 MB; use Git LFS or a Zenodo DOI for the release copy.
- The scripts still contain `/mnt/data/...` paths; make them relative before
  advertising a one-command reproduction.
- `Cohort A` is the P4 `frozen_independent2048.npz`; `Cohort B` is the P5
  `cohort_fresh1024_tuned_confirm.npz`. Confirm this mapping against the
  manuscript before release.
