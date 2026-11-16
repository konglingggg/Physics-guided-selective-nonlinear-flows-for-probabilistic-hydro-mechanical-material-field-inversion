# Archive notes

| Item | Status |
|---|---|
| C32 Biot solver | `solver/biot_mechanism_audit.py` is the production solver used for every dataset (2-D 16×16, 3-step backward Euler, material tangent). `solver/c32_linked_storage_audit.py` is the linked-storage audit. |
| C73/C74 flow code | Present under `flow/`; the C74 → C73 → C69 → C64 → C32 import chain has been made relative-path clean and tested end-to-end. |
| Frozen checkpoints | 57 `.pt` under `checkpoints/`, plus C73 dual checkpoints under `flow/c73/dual_checkpoints/`. |
| dev/validation/confirmation splits | `data/splits/*.json` (1,600 / 240 / 240 sequential indices). |
| Cohort A/B/C/D | Present as `.npz` under `data/cohorts/`. |
| Result tables | Tables 5, 6, 8, 9, 10 reconstructed exactly; Table 7 is a tidy reconstruction of the LR audit + Cohort C block. |

## Release checklist

- Checkpoints are ~60 MB; use Git LFS or a Zenodo DOI for the public release.
- Absolute authoring paths have been removed from Python, CSV and JSON files;
  `tools/check_paths.py` enforces this.
- Confirm the Cohort A = P4 `frozen_independent2048.npz` and Cohort B = P5
  `cohort_fresh1024_tuned_confirm.npz` mapping against the manuscript.
