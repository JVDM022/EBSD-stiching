# Self-Supervised EBSD Stitching: Physics, Statistics, Architecture, and Math

## Goal

The target is EDAX-level stitching reliability: stable alignment across adjacent EBSD maps, low false seam matches, physically plausible grain continuity, and auditable failure modes. The safest path is not a purely visual model. It is a physics-first system with machine learning used as a redundancy layer.

The core idea is:

1. Use EBSD orientation physics to reject impossible links.
2. Use graph structure and local statistics to rank plausible links.
3. Train the GATv2 matcher self-supervised from high-confidence consistency rules.
4. Fuse heuristic and ML scores with hard physical gates.
5. Validate on held-out maps and, when possible, parent full-map geometry.

## Why Not Use DINOv3 Alone

DINOv3-style vision features can help if there is image-like information:

- EBSD pattern images
- IQ maps
- CI maps
- IPF/RGB orientation maps
- band contrast maps
- BSE/SEM reference images

However, DINOv3 should be an auxiliary redundancy branch, not the main decision maker. EBSD stitching is constrained by crystallography and scan geometry. A vision backbone can learn texture similarity, but it does not inherently know crystal symmetry, phase compatibility, scan tilt, or misorientation limits.

Recommended use:

- Freeze a DINOv3 encoder.
- Extract dense tile descriptors or boundary-strip descriptors.
- Add DINO similarity as one extra pair feature.
- Keep hard EBSD gates unchanged.
- Let gated fusion reject visual matches that violate orientation, phase, or geometry.

Do not let a visual model override physical impossibility.

## Material Science Physics

### EBSD Orientation

Each EBSD point stores an orientation, commonly as Bunge Euler angles:

```text
phi1, Phi, phi2
```

These are converted to a rotation matrix:

```text
g = Rz(phi1) Rx(Phi) Rz(phi2)
```

Two neighboring points or grains are compared by the misorientation:

```text
Delta g = g1 g2^T
theta = arccos((trace(Delta g) - 1) / 2)
```

For production accuracy, this should be upgraded to minimum misorientation over crystal symmetry:

```text
theta_min = min_{S_i, S_j in G} angle(S_i g1 g2^T S_j)
```

where `G` is the crystal symmetry group. This matters because cubic, hexagonal, and other lattices have equivalent orientations that should not be treated as different.

### Phase Compatibility

Candidate grain links should normally require matching phase IDs. A nickel grain should not stitch to a different phase unless the workflow explicitly supports phase transformations or indexing ambiguity.

Hard gate:

```text
phase_A == phase_B
```

### Scan Geometry

EBSD maps are affected by sample tilt. A common correction stretches the scan `y` coordinate:

```text
y_corrected = y_scan / cos(tilt_angle)
```

For a typical EBSD tilt:

```text
tilt_angle = 70 deg
```

The stitching transform between adjacent tiles is estimated from accepted grain matches. The current pipeline uses affine estimation with RANSAC:

```text
p_A ~= A p_B + t
```

RANSAC rejects candidate matches that do not agree with the dominant transform.

### Grain Continuity

A correct stitch should preserve:

- orientation continuity
- grain boundary topology
- local neighbor relationships
- approximate grain size and shape
- phase identity
- spatial continuity across the seam

This is why graph features are important. Matching grains independently is weaker than matching grains plus their local boundary neighborhood.

## Statistical Model

### Candidate Link Features

For each candidate grain pair `(a, b)`, the pipeline computes:

```text
geometry_score
orientation_score
connectivity_score
quality_score
misorientation_deg
centroid_distance
CI_mean
IQ_mean
phase_match
in_expected_overlap_region
```

Example scoring logic:

```text
orientation_score = exp(-misorientation_deg / 5)
geometry_score = exp(-centroid_distance / scale)
quality_score = exp(-abs(IQ_a - IQ_b) / IQ_scale) * exp(-abs(CI_a - CI_b) / 0.25)
```

The heuristic score is a weighted combination:

```text
h = w_g geometry_score + w_o orientation_score
    + w_l local_consistency_score
```

This score is not trusted blindly. It is used to form pseudo-labels and as one input to gated fusion.

### Self-Supervised Pseudo-Labels

The system no longer needs parent overlap labels for training. It generates pseudo-labels from consistency.

Positive pseudo-label:

```text
candidate is mutual best A->B and B->A
heuristic_score >= positive_threshold
misorientation_deg <= max_positive_misorientation
centroid_distance <= max_positive_distance
phase_match == True
candidate lies in expected overlap region
```

Negative pseudo-label:

```text
heuristic_score <= negative_threshold
or misorientation_deg >= min_negative_misorientation
or centroid_distance >= min_negative_distance
or phase mismatch
or outside expected overlap
```

Ambiguous candidates receive zero weight:

```text
ssl_weight = 0
```

They are ignored by the loss.

### Weighted Binary Cross-Entropy

The GATv2 matcher predicts:

```text
p_ml = P(link is valid | graph, pair features)
```

Training uses weighted BCE:

```text
L = sum_i w_i BCE(p_i, y_i) / sum_i w_i
```

where:

```text
y_i = pseudo-label
w_i = confidence weight
```

This prevents low-confidence pseudo-labels from dominating training.

## Architecture

### Graph Construction

Each tile becomes a grain graph:

```text
node = grain or cluster
edge = neighboring grains or kNN centroid adjacency
```

Node features include:

```text
centroid_x
centroid_y
area
mean_IQ
mean_CI
orientation quaternion q0, q1, q2, q3
degree
```

Candidate pair features include:

```text
geometry_score
orientation_score
connectivity_score
quality_score
normalized_misorientation
```

### GATv2 Matcher

The model is a two-tower graph encoder:

```text
GATv2(tile A graph) -> node embeddings z_A
GATv2(tile B graph) -> node embeddings z_B
```

For candidate pair `(a, b)`:

```text
pair_embedding = concat(z_A[a], z_B[b], pair_features)
logit = MLP(pair_embedding)
p_ml = sigmoid(logit)
```

The graph attention layer lets a grain embedding depend on local boundary context, not only its own mean orientation.

## Gated Fusion

The final decision combines:

```text
h = physics/statistics heuristic score
g = GATv2 probability
```

But hard gates run first:

```text
reject if misorientation_deg > max
reject if centroid_distance > max
reject if CI_mean < min
reject if IQ_mean < min
reject if phase mismatch
reject if outside expected overlap region
```

Then fusion is conservative:

```text
if h high and g not low: use h
if g high and h not low: use g
if both low: reject
if strong disagreement: use min(h, g)
else: average h and g
```

This is the right structure for high-accuracy EBSD stitching because it reduces false positives. False seam matches are usually more damaging than missed links.

## Validation Is Still Required

Self-supervised training removes the need for manual labels. It does not remove the need for validation.

Validation is needed for:

- threshold tuning
- calibration
- detecting pseudo-label collapse
- measuring false-match rate
- comparing heuristic-only, ML-only, and gated-fusion behavior
- verifying stitch transform accuracy
- checking generalization to a different parent ANG file

The validation labels can come from:

- synthetic distortions with known transform
- cropped tiles from a known parent map
- held-out parent full ANG file
- manual review of a small seam subset
- repeat scans of the same region

The key rule:

```text
Do not use validation labels in the training loss.
```

Using labels only for evaluation and threshold selection is compatible with self-supervised training.

## Accuracy Roadmap Toward EDAX-Level Reliability

### Highest Priority

1. Use `orix` for ANG loading and crystal-map handling.
2. Add crystal symmetry to misorientation.
3. Improve grain segmentation using validated grain IDs when available.
4. Replace kNN graph adjacency with true boundary adjacency from pixel labels.
5. Keep parent full-map validation for held-out final audit.
6. Tune hard gates on validation only.

### Next Priority

1. Add seam-specific local affine or polynomial distortion correction.
2. Add uncertainty estimates for every accepted link.
3. Add robust one-to-one assignment before RANSAC.
4. Track per-phase and per-texture performance.
5. Reject seams with weak consensus rather than forcing a stitch.

### Implemented In The Notebook

The right/down stitching notebook now treats these as active reliability layers:

1. `misorientation_deg()` uses crystal-symmetry-aware disorientation, with cubic symmetry enabled by default through `PHASE_SYMMETRY`.
2. Candidate links carry `phase_a`, `phase_b`, and `phase_match`; phase mismatch is a hard gated-fusion reject.
3. `link_uncertainty` and `link_confidence` are exported for each candidate, including accepted links.
4. `robust_one_to_one_candidates()` performs Hungarian one-to-one assignment before RANSAC.
5. `validate_one_seam_transform()` rejects seams with too few matches, high uncertainty, or weak RANSAC consensus.
6. `apply_local_polynomial_correction()` optionally fits a seam-local polynomial residual correction after affine alignment.
7. `per_category_analysis()` reports phase, texture, CI, degree, misorientation, and uncertainty performance bins.

### Optional Vision Redundancy

Add a frozen vision encoder only after the EBSD physics path is strong.

Useful image inputs:

- IPF color tile
- IQ map
- CI map
- band contrast map
- EBSD pattern crops if available

Add features such as:

```text
dino_boundary_similarity
dino_dense_correspondence_score
dino_attention_overlap_score
```

Then include these in `pair_features`, but keep gated fusion hard gates.

## Failure Modes To Monitor

### Pseudo-Label Collapse

Symptom:

```text
all pseudo-labels become negative
or only one positive per many tile pairs
```

Response:

```text
relax positive thresholds slightly
increase overlap width
improve geometry normalization
inspect segmentation
```

### Texture Ambiguity

Symptom:

```text
many grains have similar orientations
```

Response:

```text
increase graph context
use boundary topology
use phase and quality gates
require stronger RANSAC consensus
```

### Bad Segmentation

Symptom:

```text
grain clusters do not correspond to physical grains
```

Response:

```text
use vendor grain IDs
use misorientation connected components
validate segmentation before link prediction
```

### Visual Shortcut Risk

Symptom:

```text
DINO or image branch matches visually similar but crystallographically impossible areas
```

Response:

```text
keep visual model auxiliary
never bypass phase, misorientation, or geometry gates
```

## Practical Recommendation

For EDAX-level accuracy, prioritize EBSD-native correctness first:

```text
orix ANG loading
crystal symmetry
validated grain segmentation
boundary graph topology
self-supervised GATv2
hard-gated fusion
held-out parent validation
```

Add DINOv3 later as redundancy if image-like map channels are available. It is most useful for difficult seams where orientation and graph signals are underdetermined, but it should remain subordinate to material physics.
