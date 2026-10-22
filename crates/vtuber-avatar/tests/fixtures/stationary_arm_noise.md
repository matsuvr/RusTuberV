The numeric fixture contains 151 shoulder-relative elbow/wrist perturbations
from the local debug capture of 2026-10-22, camera sequences 991 through 1141
(anatomical right arm, displayed left with mirroring). Each channel has its
mean removed and is divided by the mean total observed arm length,
0.43544170468717835 m. Coordinates use the tracking basis (+Y up, +Z forward).
There are no images, absolute body positions, timestamps, or model identifiers.

The runtime regression adds these residuals to an anatomically valid synthetic
pose, calibrates from that pose, then executes the production tracking filter,
goal conversion, constrained solver, checked joint path, and bone compositor.
This preserves the captured noise spectrum without requiring a private model
or treating the original noisy estimates as an anatomically valid skeleton.
