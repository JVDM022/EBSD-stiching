# Technical Director Manual: Raster + ML EBSD Stitching

## Document Control

Document: `EBSD_RASTER_ML_STITCHING_TDM.md`

System: Raster-first, self-supervised ML-assisted EBSD tile stitching

Primary notebook: `ebsd_raster_ml_stitching.ipynb`

Primary current output: `raster_ml_stitch_output/`

Parent ground-truth source used for current validation:

```text
EBSD stiching dev dataset/5%coldrolled_30min_975C_65mag2.ang
```

Current validated tile set:

```text
Cropped/
```

Current validation headline:

```text
Parent ANG selected-shift accuracy: 137 / 137
Mean selected x error: 0 px
Mean selected y error: 0 px
Max selected x error: 0 px
Max selected y error: 0 px
```

## Executive Summary

This system stitches EBSD `.ang` tile maps by treating EBSD stitching as a raster registration and seam-selection problem. The pipeline does not reconstruct grains first. It reads the tiled EBSD maps, gridifies each tile into raster arrays, evaluates candidate right/down neighbor translations, scores overlap consistency using EBSD physics and image-like map agreement, trains a small self-supervised logistic model from high-confidence pseudo-labels, selects one best shift per neighbor seam, solves global tile origins, and writes a stitched IQ/CI/IPF mosaic.

The most important validation is not a candidate-level confusion matrix. Stitching is not primarily a binary classification problem over every possible candidate shift. It is a top-1 selection and ranking problem:

```text
For each neighboring tile pair:
    score all plausible candidate shifts
    select the best shift
    validate selected shift against parent ANG ground truth
```

The current direct parent-ANG validation confirms that every selected seam shift matches the parent map coordinate truth:

```text
n_pairs: 137
selected_top1_accuracy: 1.0
best_score_top1_accuracy: 1.0
top1_true_shift_accuracy: 1.0
mean_true_rank: 1.0
mean_reciprocal_rank: 1.0
mean_abs_selected_x_error_px: 0.0
mean_abs_selected_y_error_px: 0.0
```

This is a strong result for a controlled cropped-parent experiment. It is not yet sufficient to claim EDAX/OIM production-level reliability across all real acquisition conditions. The current result proves that the algorithm can recover the true parent-map geometry for this clean cropped dataset. Production reliability requires additional tests on separately acquired maps, distorted tiles, different materials, multi-phase datasets, variable pattern quality, and cases without a parent-coordinate reference.

## System Goal

The goal is to produce an auditable EBSD stitching workflow that:

1. Aligns adjacent EBSD tiles at pixel-level accuracy.
2. Uses EBSD physical consistency instead of pure image similarity.
3. Provides clear confidence and rejection behavior.
4. Produces validation artifacts that separate seam quality from ground-truth placement accuracy.
5. Can be extended toward production-grade stitching after broader validation.

The desired operating mode is:

```text
Input: directory of clean cropped .ang tiles
Output: stitched mosaic arrays, preview images, selected seam shifts, validation reports
```

The system is intentionally raster-first. Grain reconstruction can be done after stitching the map. This avoids a common failure mode where grain segmentation differences near tile edges cause unstable pre-stitch grain matching.

## Repository Components

### Main Notebook

```text
ebsd_raster_ml_stitching.ipynb
```

Responsibilities:

- Reads `.ang` tiles.
- Converts unordered EBSD rows into raster tile arrays.
- Computes orientation math under FCC symmetry.
- Builds candidate shift sets for adjacent right/down tile pairs.
- Scores candidate shifts using IQ, CI, valid-pixel fraction, and misorientation features.
- Trains a self-supervised logistic-regression seam scorer.
- Selects the best seam shift per adjacent tile pair.
- Solves global tile origins.
- Writes stitched mosaic images and arrays.
- Runs pseudo-label diagnostics.
- Runs direct parent-ANG ground-truth validation as top-1/ranking validation.

### Input Tile Directory

```text
Cropped/
```

Important files:

```text
Cropped/tile_manifest.csv
Cropped/tile_r{row}_c{col}_clean.ang
Cropped/tile_r{row}_c{col}_distorted.ang
```

The current stitching run uses clean tiles ending in:

```text
_clean.ang
```

### Parent Ground-Truth File

```text
EBSD stiching dev dataset/5%coldrolled_30min_975C_65mag2.ang
```

This parent map is used only for validation. It is not needed for normal stitching. Because the clean tiles were cropped from this file, the parent map provides exact coordinate truth.

### Output Directory

```text
raster_ml_stitch_output/
```

Core output files:

```text
candidate_shift_scores.csv
selected_pair_shifts.csv
tile_origins.csv
stitched_mosaic_arrays.npz
stitched_iq.png
stitched_ci.png
stitched_ipf_preview.png
stitch_summary.json
```

Validation files:

```text
raster_ml_stitch_output/validation/validation_summary.json
raster_ml_stitch_output/validation/parent_ang_ranking_summary.json
raster_ml_stitch_output/validation/parent_ang_pair_ranking_validation.csv
raster_ml_stitch_output/validation/parent_ang_selected_accuracy_by_direction.csv
raster_ml_stitch_output/validation/parent_ang_selected_pair_shifts.csv
raster_ml_stitch_output/validation/parent_ang_selected_shift_error.png
raster_ml_stitch_output/validation/parent_ang_true_rank_histogram.png
raster_ml_stitch_output/validation/parent_ang_score_margin_by_direction.png
```

## High-Level Architecture

The architecture is a staged pipeline:

```text
.ang tiles
    |
    v
ANG numeric parser
    |
    v
MTEX-style raster gridify
    |
    v
Right/down adjacency from tile manifest
    |
    v
Overlap strip extraction
    |
    v
Candidate shift generation
    |
    v
Candidate feature extraction
    |
    v
Classical physics score + pseudo-label
    |
    v
Self-supervised logistic ML scorer
    |
    v
Final score fusion
    |
    v
Best candidate selection per seam
    |
    v
Global tile origin solve
    |
    v
Mosaic construction
    |
    v
Validation and reporting
```

The key design decision is that the model does not directly predict the global mosaic. It scores local seam candidates. The global mosaic is then assembled from selected local relationships.

## Data Model

### TileGrid

Each tile is represented as a raster grid:

```text
TileGrid:
    name
    row
    col
    path
    x_values
    y_values
    x_step
    y_step
    phi1
    Phi
    phi2
    iq
    ci
    phase
    valid
```

The raster arrays have shape:

```text
height x width
```

Where:

- `phi1`, `Phi`, `phi2` are Bunge Euler angles.
- `iq` is image quality.
- `ci` is confidence index.
- `phase` is currently forced to single-phase FCC for this SCC 316 test.
- `valid` marks finite, usable EBSD points.

### PairCandidate

Each candidate seam shift is represented as:

```text
PairCandidate:
    pair_id
    direction
    tile_a
    tile_b
    shift_y
    shift_x
    origin_dy
    origin_dx
    iq_corr
    ci_corr
    valid_fraction
    median_misorientation_deg
    p90_misorientation_deg
    overlap_pixels
    candidate_source
    classical_score
    pseudo_label
    ml_probability
    final_score
    selected
    accepted
```

The most important fields for stitching are:

```text
origin_dy, origin_dx
final_score
selected
accepted
```

The most important fields for quality control are:

```text
median_misorientation_deg
p90_misorientation_deg
iq_corr
ci_corr
valid_fraction
```

## ANG Loading and Gridification

### ANG Numeric Rows

The `.ang` file is read as numeric rows. Common columns are:

```text
phi1 Phi phi2 x y IQ CI phase ...
```

The notebook preserves the numeric data needed for raster alignment and orientation calculations.

### Radian/Degree Detection

Euler angles are expected to be radians. If the maximum Euler-angle value appears degree-like, the notebook converts to radians:

```text
if max(phi1, Phi, phi2) > 2*pi:
    angles = deg2rad(angles)
```

### MTEX-Style Gridify

EBSD maps can be rectangular or hex/staggered. A naive grid using all unique x and y values can create sparse arrays for hex-like exports. The notebook detects this and uses scan-row gridification when appropriate.

It computes:

```text
rectangular_fill = n_points / (n_unique_x * n_unique_y)
row_counts = number of points per y row
use_scan_rows = rectangular_fill < 0.75 and row_counts has <= 3 unique counts
```

When `use_scan_rows` is true:

1. Rows are grouped by y.
2. Points inside each row are sorted by x.
3. A fixed scan-row width is selected from the modal row count.
4. Arrays are filled by row order and column order.
5. Effective x positions are estimated as the median x per scan column.

This matters for the current parent ANG because:

```text
raw unique x step can be 0.5
effective scan-column x step is 1.0
```

Using the wrong x step makes ground-truth validation appear wrong. The notebook uses the scan-row-consistent step.

## Coordinate Systems

The system uses three related coordinate systems.

### Parent ANG Coordinates

The parent `.ang` stores physical scan coordinates:

```text
x_parent
y_parent
```

For the current parent file:

```text
parent_x_step = 1.0
parent_y_step_raw ~= 0.86603
```

### Tilt-Corrected Coordinates

The crop script applies y-tilt correction:

```text
y_corrected = y_parent / cos(tilt_angle)
```

For the current run:

```text
tilt_angle = 70 deg
tilt_factor = 1 / cos(70 deg) ~= 2.9238044
parent_y_step_after_tilt ~= 2.5321023
```

This is why direct parent validation must account for the same tilt correction. If it does not, vertical offsets will not match.

### Stitcher Pixel Coordinates

The stitcher compares raster arrays. A seam offset is stored in pixel units:

```text
origin_dx = tile_b_origin_x_px - tile_a_origin_x_px
origin_dy = tile_b_origin_y_px - tile_a_origin_y_px
```

The conversion from parent coordinates to pixel offsets is:

```text
gt_dx = round((x_min_b - x_min_a) / parent_x_step)
gt_dy = round((y_min_b - y_min_a) / parent_y_step_after_tilt)
```

For the current parent-ANG validation, the selected stitch offsets match these ground-truth offsets exactly:

```text
selected_origin_dx - gt_dx = 0
selected_origin_dy - gt_dy = 0
```

## EBSD Orientation Math

### Bunge Euler Angles

The EBSD orientation is represented by Bunge Euler angles:

```text
phi1, Phi, phi2
```

The rotation matrix is:

```text
g = Rz(phi1) Rx(Phi) Rz(phi2)
```

Expanded components are implemented in the notebook to avoid dependency on an external crystallography library for this stage.

### Misorientation

For two orientations:

```text
g_a
g_b
```

The relative rotation is:

```text
delta = g_a * transpose(g_b)
```

The rotation angle is:

```text
theta = arccos((trace(delta) - 1) / 2)
```

Numerically:

```text
cos_theta = clip((trace(delta) - 1) / 2, -1, 1)
theta = arccos(cos_theta)
```

### FCC Symmetry

For cubic/FCC materials, many rotations are crystallographically equivalent. The physically meaningful misorientation is the minimum angle over the cubic symmetry group:

```text
theta_min = min over S in cubic_symmetry of angle(S * g_a * transpose(g_b))
```

The notebook uses FCC symmetry-aware disorientation for seam scoring. This is essential. Without symmetry, physically identical orientations can be scored as different.

## Candidate Shift Generation

The system only compares adjacent right/down pairs from the tile manifest:

```text
horizontal: tile(r, c) -> tile(r, c + 1)
vertical:   tile(r, c) -> tile(r + 1, c)
```

For each pair, the expected overlap width/height is inferred from manifest crop bounds when available. Otherwise, it uses a fallback fraction of the tile dimension.

### Overlap Strip Extraction

For a horizontal pair:

```text
strip_a = right side of tile_a
strip_b = left side of tile_b
```

For a vertical pair:

```text
strip_a = bottom side of tile_a
strip_b = top side of tile_b
```

### Candidate Offsets

The candidate set includes:

1. Zero/nominal offsets.
2. Search-radius perturbations around the nominal offset.
3. IQ-based phase-correlation candidates when available.

In the current configured run:

```text
search_radius = 1
```

This produces local candidates such as:

```text
shift_y in {-1, 0, 1}
shift_x in {-1, 0, 1}
```

For each candidate, the notebook computes the implied global relative origin:

```text
origin_dx = nominal_dx + shift_x
origin_dy = nominal_dy + shift_y
```

## Candidate Feature Extraction

Each candidate shift is evaluated on the overlapping valid pixels.

### IQ Correlation

IQ correlation measures whether the two overlap strips have matching image-quality structure:

```text
iq_corr = corr(IQ_a_overlap, IQ_b_overlap)
```

If the overlap is correct, grain and pattern-quality structures tend to align.

### CI Correlation

CI correlation measures confidence-index continuity:

```text
ci_corr = corr(CI_a_overlap, CI_b_overlap)
```

This is useful but can be less reliable than IQ or orientation continuity depending on indexing quality.

### Valid Fraction

The valid fraction is the ratio of usable overlap pixels:

```text
valid_fraction = n_valid_overlap_pixels / n_overlap_pixels
```

Low valid fraction indicates insufficient evidence for a seam.

### Median Misorientation

The primary orientation-consistency statistic is:

```text
median_misorientation_deg = median(theta_min over sampled overlap pixels)
```

The notebook also computes:

```text
p90_misorientation_deg
```

These detect seam shifts that align IQ or CI structure but violate crystallographic continuity.

### Shift Magnitude

The model includes a shift penalty feature:

```text
shift_abs = abs(shift_y) + abs(shift_x)
```

The purpose is to prefer simpler offsets when two candidates have similar seam evidence.

## Classical Seam Score

The classical score is a deterministic physics/statistics score. It combines:

```text
IQ agreement
CI agreement
valid overlap fraction
low FCC misorientation
small shift preference
```

A conceptual form is:

```text
classical_score =
    w_iq * f_iq(iq_corr)
  + w_ci * f_ci(ci_corr)
  + w_valid * valid_fraction
  + w_mis * f_mis(median_misorientation_deg, p90_misorientation_deg)
  + w_shift * f_shift(shift_abs)
```

Where:

```text
f_iq increases as iq_corr increases
f_ci increases as ci_corr increases
f_mis decreases as misorientation increases
f_shift decreases as shift_abs increases
```

The exact implementation is in the notebook candidate scoring cell.

## Self-Supervised Learning

### Why Self-Supervision

The model does not require manual seam labels. It generates pseudo-labels from high-confidence EBSD seam rules.

The pseudo-label rule in the current run is:

```text
pseudo-positive when:
    valid_fraction >= 0.60
    IQ_corr >= 0.45
    median_FCC_misorientation <= 8 deg
```

Otherwise the candidate is pseudo-negative.

### Important Meaning of Pseudo-Positive

A pseudo-positive is not the same as ground truth.

```text
pseudo-positive = "this seam candidate looks physically plausible by heuristic rules"
ground-truth positive = "this candidate matches the known parent ANG offset"
```

The model is trained to recognize plausible seam quality. It is not trained directly as an exact-offset classifier.

### Logistic Regression Model

The ML model is logistic regression with class balancing:

```text
p_ml = sigmoid(beta_0 + beta^T x)
```

Feature vector:

```text
x = [
    iq_corr,
    ci_corr,
    valid_fraction,
    median_misorientation_deg,
    p90_misorientation_deg,
    shift_abs,
    classical_score
]
```

For the current run, the fitted coefficients are:

```text
iq_corr: 3.4099169385
ci_corr: -0.7410768854
valid_fraction: -0.2092256368
median_misorientation_deg: -1.6713949666
p90_misorientation_deg: -1.3990339638
shift_abs: -0.0902699430
classical_score: 2.2466644803
intercept: 10.8830622989
```

Interpretation:

- Higher IQ correlation increases seam probability.
- Larger misorientation strongly decreases seam probability.
- Larger shift slightly decreases seam probability.
- Higher classical score increases seam probability.
- CI correlation has a negative coefficient in this fitted dataset, which likely reflects feature collinearity or dataset-specific behavior. It should not be overinterpreted as a universal physical rule.

### Train/Validation Split

The self-supervised split is by seam pair, not by individual candidate only:

```text
train_fraction_by_pair = 0.70
n_train_pairs = 96
n_validation_pairs = 41
```

Current pseudo-label validation:

```text
validation n = 381
validation positives = 369
validation negatives = 12
validation ROC AUC ~= 0.99977
validation PR AUC ~= 0.99999
```

This confirms that the self-supervised scorer agrees with the pseudo-label rule. It does not replace parent-ANG ground-truth validation.

## Score Fusion

The final score combines ML probability and classical score:

```text
final_score = 0.65 * ml_probability + 0.35 * classical_score
```

This is a pragmatic fusion:

- ML score captures the learned combination of features.
- Classical score preserves deterministic physics behavior.
- The final score remains interpretable and robust to small model issues.

The selected shift for each seam is the highest-scoring candidate after scoring and filtering.

## Seam Acceptance

A selected seam is marked accepted only if it passes hard thresholds:

```text
final_score >= min_accept_score
median_misorientation_deg <= max_accept_misorientation
valid_fraction >= min_valid_fraction
```

Current settings:

```text
min_accept_score = 0.65
max_accept_misorientation = 8.0 deg
min_valid_fraction = 0.60
```

This produces:

```text
n_selected_pairs = 137
n_accepted_seams = 135
accepted_seam_fraction = 0.9854014599
```

Important: A seam can be spatially correct but rejected by seam-quality rules. In the current run:

```text
parent-ANG-correct selected seams = 137
accepted parent-correct seams = 135
rejected parent-correct seams = 2
```

The two rejected seams are correct in position but fail the quality rule because their overlap misorientation is high. This is a quality rejection, not a spatial-placement failure.

## Global Tile Origin Solve

Each selected pair defines a relative constraint:

```text
origin(tile_b) - origin(tile_a) = (origin_dy, origin_dx)
```

The system solves tile origins across the connected grid. Conceptually:

```text
For each accepted/selected edge e = (a, b):
    y_b - y_a = dy_e
    x_b - x_a = dx_e
```

This is a graph embedding problem over tile positions. In a simple grid with consistent edges, origins can be propagated from the anchor tile. In more complex cases, a least-squares graph solve can minimize residuals:

```text
min over origins sum_e ||(origin_b - origin_a) - delta_e||^2
```

The current output is:

```text
tile_origins.csv
```

Current origin-grid validation:

```text
n_tiles_compared = 78
x_step_px = 160
y_step_px = 63
mean_abs_origin_dx_error_px = 0
mean_abs_origin_dy_error_px = 0
max_abs_origin_dx_error_px = 0
max_abs_origin_dy_error_px = 0
```

## Mosaic Construction

After tile origins are solved, the notebook writes:

```text
stitched_iq.png
stitched_ci.png
stitched_ipf_preview.png
stitched_mosaic_arrays.npz
```

The mosaic arrays include:

```text
iq
ci
ipf_rgb
source
```

The source array records tile source information for pixels where multiple tiles overlap.

## Validation Philosophy

The validation is split into two categories:

1. Pseudo-label seam diagnostics.
2. Direct parent-ANG ground-truth placement validation.

These answer different questions.

### Pseudo-Label Validation

Question:

```text
Does the scorer reproduce the self-supervised seam-quality rule?
```

This is useful for:

- Checking calibration of seam score.
- Detecting whether ML training failed.
- Measuring consistency of IQ/CI/misorientation features.

It is not the final proof of spatial correctness.

### Parent-ANG Ground-Truth Validation

Question:

```text
Did the stitcher select the true parent-map shift for each seam?
```

This is the main validation for cropped-parent experiments.

The notebook intentionally does not use parent candidate-level threshold confusion as the main result, because it asks a misleading question:

```text
If every candidate is thresholded independently, does only the exact shift score positive?
```

That punishes near-shift candidates that score well but are never selected. Stitching is a ranking problem, not a threshold-every-candidate problem.

## Direct Parent-ANG Ground Truth Method

The parent ANG file stores the true coordinate grid. The cropped tiles preserve scan coordinates after the y-tilt correction. The validation computes:

```text
parent_x_step
parent_y_step_after_tilt
tile_a_min_x, tile_a_min_y
tile_b_min_x, tile_b_min_y
```

The ground-truth relative offset is:

```text
gt_dx = round((tile_b_min_x - tile_a_min_x) / parent_x_step)
gt_dy = round((tile_b_min_y - tile_a_min_y) / parent_y_step_after_tilt)
```

The selected stitch error is:

```text
error_x = selected_origin_dx - gt_dx
error_y = selected_origin_dy - gt_dy
```

The selected seam is correct if:

```text
abs(error_x) <= tolerance_px
abs(error_y) <= tolerance_px
```

Current tolerance:

```text
tolerance_px = 0
```

Current result:

```text
error_x = 0 for all selected seams
error_y = 0 for all selected seams
```

This proves that the selected shifts match the parent ANG coordinate truth exactly for this cropped dataset.

## Ground-Truth Ranking Metrics

The notebook writes:

```text
parent_ang_ranking_summary.json
parent_ang_pair_ranking_validation.csv
parent_ang_selected_accuracy_by_direction.csv
```

### Selected Top-1 Accuracy

Definition:

```text
selected_top1_accuracy =
    number of selected seams matching parent ANG truth / number of selected seams
```

Current value:

```text
1.0
```

### Best Score Top-1 Accuracy

Definition:

```text
best_score_top1_accuracy =
    number of seams where highest final_score candidate is the parent-true candidate / number of seams
```

Current value:

```text
1.0
```

This means the scoring itself ranks the exact parent shift first.

### True Shift Rank

Definition:

```text
true_rank = rank of parent-true candidate when candidates are sorted by final_score descending
```

Current values:

```text
mean_true_rank = 1.0
median_true_rank = 1.0
```

### Top-K Accuracy

Definition:

```text
topK_true_shift_accuracy =
    fraction of seams where parent-true candidate rank <= K
```

Current values:

```text
top1_true_shift_accuracy = 1.0
top2_true_shift_accuracy = 1.0
top3_true_shift_accuracy = 1.0
top5_true_shift_accuracy = 1.0
```

### Mean Reciprocal Rank

Definition:

```text
MRR = mean(1 / true_rank)
```

Current value:

```text
mean_reciprocal_rank = 1.0
```

### Score Margin

Definition:

```text
true_minus_best_false_margin =
    final_score(parent_true_candidate) - final_score(best_non_true_candidate)
```

Current values:

```text
mean_true_minus_best_false_margin ~= 0.06008947
min_true_minus_best_false_margin ~= 0.02302018
```

The positive minimum margin means the true candidate outranks the strongest false candidate for every seam. The minimum margin is not huge, so margin monitoring is valuable for harder datasets.

## Current Main Results

### Dataset Scale

```text
n_tiles = 78
n_pairs = 137
n_candidates = 1287
n_selected_pairs = 137
n_accepted_seams = 135
```

### Parent ANG Ground Truth

```text
n_pairs = 137
selected_top1_accuracy = 1.0
best_score_top1_accuracy = 1.0
mean_abs_selected_x_error_px = 0.0
mean_abs_selected_y_error_px = 0.0
max_abs_selected_x_error_px = 0.0
max_abs_selected_y_error_px = 0.0
mean_true_rank = 1.0
top1_true_shift_accuracy = 1.0
```

### Direction Breakdown

Horizontal:

```text
n_pairs = 65
selected_correct = 65
mean_abs_x_error_px = 0
mean_abs_y_error_px = 0
selected_accuracy = 1.0
accepted = 64
```

Vertical:

```text
n_pairs = 72
selected_correct = 72
mean_abs_x_error_px = 0
mean_abs_y_error_px = 0
selected_accuracy = 1.0
accepted = 71
```

### Seam Quality

```text
accepted_seam_fraction = 0.9854014599
median selected IQ correlation = 1.0
median selected CI correlation = 1.0
median selected misorientation = 0.0 deg
median selected valid fraction = 1.0
```

## Why DINOv3 Is Not Needed for This Ground Truth

DINOv3 is a visual feature encoder. It could compare images or image-like maps. It is not needed for the current parent-ANG validation because direct coordinate truth is stronger.

The current ground-truth validation uses:

```text
actual parent ANG coordinates
tile coordinate minima
parent x/y scan spacing
tilt correction
selected stitch offsets
```

This directly measures whether the selected stitch offset equals the known parent-map offset. No visual embedding can provide stronger truth than the original coordinate system.

DINOv3 may be useful later when:

- There is no parent ANG ground truth.
- Tiles are acquired separately and coordinates are unreliable.
- There are image-like EBSD maps where semantic texture helps.
- IQ/CI/orientation evidence is ambiguous.

Recommended role for DINOv3:

```text
DINOv3 similarity = auxiliary feature
not a replacement for EBSD physics
not allowed to override phase/orientation impossibility gates
```

Possible future feature:

```text
final_features += [
    dino_strip_similarity,
    dino_dense_patch_similarity,
    dino_margin_best_vs_second
]
```

But for the current cropped-parent validation, DINOv3 would be unnecessary and weaker than coordinate truth.

## Reliability Assessment

### What Is Proven

For the current SCC 316 cropped-parent dataset:

```text
The selected tile-to-tile shifts exactly match the parent ANG coordinate truth.
```

This means the ML-assisted stitcher is spatially correct on this controlled dataset.

### What Is Not Proven

This does not yet prove EDAX-level production reliability for:

- separately acquired EBSD scans,
- stage drift,
- scan rotation,
- nonlinear detector distortion,
- multi-phase materials,
- strong pattern-quality gradients,
- low-confidence indexing,
- missing overlap,
- maps without a parent coordinate reference.

### Current Maturity Level

Current state:

```text
Strong controlled proof-of-concept
```

Not yet:

```text
fully production-qualified EDAX-level stitcher
```

To move toward production qualification, run:

1. Clean cropped parent tests.
2. Distorted cropped parent tests.
3. Different material tests, including `Cropped_alloy718`.
4. Real independently acquired tile tests.
5. Failure-mode tests with insufficient overlap and incorrect neighbor pairs.
6. Cross-validation over multiple maps.
7. Comparison against EDAX/OIM exported stitched references where available.

## Operating Procedure

### Configure

In the notebook:

```python
TILES_FOLDER = "Cropped"
OUTPUT_DIR = "raster_ml_stitch_output"
```

Key parameters:

```python
search_radius = 1
orientation_samples = 120
use_ml_training = True
self_supervised_train_fraction = 0.70
min_accept_score = 0.65
max_accept_misorientation = 8.0
min_valid_fraction = 0.60
```

### Run Stitching

Run:

```python
run(args)
```

Expected primary outputs:

```text
candidate_shift_scores.csv
selected_pair_shifts.csv
tile_origins.csv
stitched_iq.png
stitched_ci.png
stitched_ipf_preview.png
stitch_summary.json
```

### Run Validation

Run the validation section:

```text
Validation: Parent ANG Ground Truth
```

This section is self-contained inside the notebook.

It writes:

```text
validation_summary.json
parent_ang_ranking_summary.json
parent_ang_pair_ranking_validation.csv
parent_ang_selected_accuracy_by_direction.csv
parent_ang_selected_shift_error.png
parent_ang_true_rank_histogram.png
parent_ang_score_margin_by_direction.png
```

### Interpret

Use these as headline metrics:

```text
selected_top1_accuracy
mean_abs_selected_x_error_px
mean_abs_selected_y_error_px
max_abs_selected_x_error_px
max_abs_selected_y_error_px
mean_true_rank
top1_true_shift_accuracy
min_true_minus_best_false_margin
```

Do not use candidate-level threshold confusion against parent ANG as the headline. It has been removed because it is not the correct task formulation.

## Failure Modes and Diagnostics

### High Misorientation but Correct Placement

Observed in current run:

```text
2 selected seams are parent-correct but rejected by seam-quality acceptance.
```

Meaning:

```text
spatial placement is correct
quality gate rejects seam due to high misorientation
```

This can happen near boundaries, indexing changes, or regions with poor orientation continuity.

### Low Score Margin

If:

```text
true_minus_best_false_margin near 0
```

Then:

```text
true shift barely outranks alternative shift
```

This is a warning for harder datasets. It does not mean failure, but it suggests lower confidence.

### Nonzero Selected Error

If:

```text
selected_error_x_px != 0
or selected_error_y_px != 0
```

Then inspect:

```text
parent_ang_pair_ranking_validation.csv
parent_ang_selected_pair_shifts.csv
candidate_shift_scores.csv
```

Check:

- whether the parent tilt correction is wrong,
- whether x/y step inference is wrong,
- whether tile coordinates were transformed,
- whether a distorted tile was used instead of clean,
- whether search radius was too small,
- whether the true shift was not in the candidate set.

### Parent Coordinate Mismatch

If parent-ANG validation does not match but mosaic visually looks correct, verify:

```text
PARENT_ANG
PARENT_TILT_DEGREES
TILES_FOLDER
whether tile coordinates are parent-global or locally reset
whether cropped tiles are clean or distorted
```

The current clean tiles preserve parent coordinates after tilt correction.

## Recommended Next Development Steps

### 1. Distorted-Tile Validation

Run on distorted tiles:

```text
tile_r*_c*_distorted.ang
```

Goal:

```text
recover parent offsets despite projective/trapezoid coordinate distortions
```

Expected additional outputs:

```text
distortion recovery error
selected shift accuracy
failure/rejection rate
```

### 2. Alloy 718 Validation

Run the same validation on:

```text
Cropped_alloy718/
```

Important:

- Confirm phase/symmetry assumptions.
- If material is FCC nickel-based alloy, FCC symmetry may remain appropriate.
- Recheck parent ANG file path for Alloy 718.

### 3. Multi-Map Qualification

Create a validation table:

```text
map_id
material
n_tiles
n_pairs
selected_top1_accuracy
mean_abs_x_error_px
mean_abs_y_error_px
min_score_margin
accepted_fraction
failure_notes
```

### 4. Production Confidence Model

Add confidence categories:

```text
green:
    selected error = 0 in validation
    score margin high
    accepted seam

yellow:
    selected error <= 1 px
    low score margin
    accepted seam

red:
    rejected seam
    inconsistent global origin
    low valid fraction
    high misorientation
```

### 5. Optional DINOv3 Auxiliary Branch

Only after physics-first validation is stable:

```text
dino_similarity = cosine(DINO(strip_a), DINO(strip_b))
```

Use as:

```text
extra feature in ML scorer
```

Do not use as:

```text
replacement for EBSD misorientation or phase checks
```

## Acceptance Criteria for This Dataset

The current dataset passes the following controlled validation criteria:

```text
selected_top1_accuracy == 1.0
best_score_top1_accuracy == 1.0
mean_abs_selected_x_error_px == 0.0
mean_abs_selected_y_error_px == 0.0
max_abs_selected_x_error_px == 0.0
max_abs_selected_y_error_px == 0.0
top1_true_shift_accuracy == 1.0
mean_true_rank == 1.0
```

Two parent-correct seams are rejected by acceptance gates:

```text
accepted_parent_correct = 135
rejected_parent_correct = 2
```

This should be reported as:

```text
spatial stitching correct
seam-quality gate rejected 2 spatially correct but high-misorientation seams
```

## Bottom Line

For the current controlled SCC 316 cropped-parent experiment, the ML-assisted raster stitcher selects the correct parent-ANG shift for every seam. The correct validation metric is selected top-1 ground-truth accuracy, not candidate-level parent threshold confusion.

The current result supports:

```text
The stitching placement is correct on this dataset.
```

It does not yet support:

```text
This is EDAX-level reliable for all EBSD acquisition conditions.
```

The next technical priority is broader validation on distorted tiles, Alloy 718, independent acquisitions, and deliberate failure cases.
