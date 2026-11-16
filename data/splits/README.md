# Splits

`development.json`, `validation.json`, `confirmation.json` contain sequential
indices for the C64/C69 dataset (1,600 / 240 / 240, total 2,080), matching the
`train` / `val` / `test` counts stored in `flow/c69/joint_dataset.npz`.
The cohort `.npz` files (A/B/C/D) do not store per-case split indices; they are
independent confirmation sets and are used whole, as described in Table 4.
