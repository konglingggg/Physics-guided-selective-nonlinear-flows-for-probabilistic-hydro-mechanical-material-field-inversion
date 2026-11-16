# C32 Biot solver / material tangent

- `biot_mechanism_audit.py` — independent 2-D square (16×16 nodal fields),
  monolithic backward-Euler (3 steps) Biot implementation with log k / log E
  material dependence; builds the coupled residual and material tangent.
- `c32_linked_storage_audit.py` — linked-storage audit for the C32 configuration.

The original C32 production driver used for the paper is not part of the
archived evidence; these two files are the archived mechanism implementation.
