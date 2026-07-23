# LinkedIn Media Files

Recommended upload order:

1. `00_linkedin_cover.png`
   - Cover graphic for the project. Use this as the first media item.

2. `01_stitched_ipf_map.png`
   - Main stitched EBSD IPF orientation map. This is the strongest technical visual.

3. `02_stitched_iq_map.png`
   - Stitched image-quality map showing intensity continuity across the mosaic.

4. `03_tile_layout_before_stitching.png`
   - Original cropped tile layout before stitching.

5. `04_validation_parent_vs_stitched_ipf.png`
   - Controlled validation comparison between stitched map and parent map.

6. `05_seam_acceptance_by_direction.png`
   - Diagnostic plot showing accepted seam fraction by direction.

7. `06_misorientation_by_direction.png`
   - Diagnostic plot showing selected seam misorientation by direction.

8. `07_app_interface_preview.png`
   - App-style interface preview for showing that the workflow can be used without reading source code.

Suggested LinkedIn caption:

Built a physics-guided EBSD tile stitching workflow for reconstructing large-area microstructure maps from overlapping `.ang` scan tiles. The tool combines image registration, IQ/CI overlap correlation, crystallographic misorientation scoring, and a lightweight self-supervised seam scorer. It supports FCC/HCP symmetry-aware checks and includes an app-style interface for no-code use.

Note:

The parent-vs-stitched validation image comes from the controlled SCC/FCC cropped-parent dataset. For Nathan's no-parent HCP case, use seam consistency, visual alignment, IQ/CI continuity, and HCP misorientation diagnostics rather than claiming ground-truth accuracy.
