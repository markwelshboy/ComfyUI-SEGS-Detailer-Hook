# ComfyUI SEGS Detailer Hook — Multi-Mode Color Match

A ComfyUI custom node for [Impact Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack) **Detailer (SEGS)** and **FaceDetailer**. Compare **six color-matching methods** without changing the detector, crop, seed or diffusion settings.

The hook saves the **original crop** in Impact Pack's `post_upscale()`, corrects the decoded result in `post_decode()`, and returns the corrected crop for Impact Pack's normal masked compositing. No Impact Pack modifications or separate crop-reference image are required.

## Install / update

```bash
cd /opt/ComfyUI/custom_nodes
git clone https://github.com/markwelshboy/ComfyUI-SEGS-Detailer-Hook.git
# Restart ComfyUI
```

For an existing installation:

```bash
cd /opt/ComfyUI/custom_nodes/ComfyUI-SEGS-Detailer-Hook
git pull --ff-only
# Restart ComfyUI, then refresh the browser
```

Requires Impact Pack, PyTorch, OpenCV and NumPy (normally already installed with Impact Pack). The hook imports Impact Pack's hook base at run time, so node startup order does not matter.

## Connecting it

Search for **Color Match Detailer Hook (Multi-Mode)** under `ImpactPack/Detailer/Hooks`. Connect its `DETAILER_HOOK` output to **Detailer (SEGS)** or **FaceDetailer**'s `detailer_hook` input.

```text
Image + SEGS ---> Detailer (SEGS) ---> Final composite
                       ^
                       | detailer_hook
              Color Match Detailer Hook (Multi-Mode)
```

The **node type ID remains `ColorMatchDetailerHookProvider`**, so previous workflows remain compatible. The original five widgets are in their original order, followed by `method` and `boundary_width`; the original LAB-chroma behavior remains the default.

## Modes

| `method` | Description | Typical use |
| --- | --- | --- |
| `lab_chroma` **(default)** | Match LAB a/b mean and capped standard deviation, without changing lightness by default | Reliable baseline for skin/face color casts |
| `lab_full` | Match LAB L/a/b mean and capped standard deviation; **always includes luminance** | Fix both color and exposure mismatch |
| `chroma_mean_only` | Adjust LAB a/b *mean* without changing chroma spread | Gentler skin-tone correction that preserves variation |
| `boundary_chroma` | Use an **inner strip along the mask boundary**, with a dominant-color similarity filter, to estimate LAB a/b mean shifts | Experiment for seams along face/skin boundaries |
| `rgb_mean_std` | RGB channel-wise mean/std matching | Baseline comparison, can change brightness |
| `rgb_histogram` | RGB channel-wise quantile/histogram matching | Experimental, potentially aggressive; use low strength |

All modes compare corresponding pixels **in the same original and regenerated crop**, not the full image. A cropped BBOX mask is rectangular—not a skin mask. The boundary mode's similarity filter is a heuristic, **not a skin segmentation model**, and cannot guarantee that hair or clothing are excluded. If a boundary ring has too few reliable samples, it falls back to masked-crop statistics.

## Settings

| Parameter | Default | Description |
| --- | --- | --- |
| `enabled` | `true` | Bypass without removing the hook |
| `strength` | `0.50` | Blend 0 (uncorrected) to 1 (fully color matched) |
| `reference_region` | `masked` | `masked` statistics only inside the mask, or `whole_crop` |
| `match_luminance` | `false` | Include lightness in LAB modes (except `lab_full`, which always does). Ignored in RGB modes |
| `mask_erosion` | `5` | Exclude edge pixels from the statistics (working crop pixels) |
| `method` | `lab_chroma` | Select one of the six matching methods |
| `boundary_width` | `24` | Approximate inner boundary band width, **in working-crop pixels**; used only by `boundary_chroma` |

**For `boundary_chroma`, use `reference_region = masked` and `noise_mask = enabled` on the detailer.** Without a mask, boundary mode becomes whole-crop mean-only LAB matching. `reference_region = whole_crop` bypasses boundary selection.

The hook sees the detailer's `noise_mask` (if enabled) in `post_upscale`. `noise_mask_feather` is Impact Pack's separate diffusion-mask setting; setting it above zero may enable differential diffusion. Use `noise_mask_feather = 0` when isolating color-match differences.

## Suggested A/B comparison

Use the same source image, BBOX detection, SEGS crop, seed, steps, denoise and all other detailer settings for each method. Start with:

- `reference_region = masked`
- `match_luminance = false`
- `mask_erosion = 5`
- `strength = 0.40`
- `boundary_width = 24`

Try `lab_chroma` vs `chroma_mean_only` vs `boundary_chroma` first. Use `lab_full` when the patch is darker/brighter, and `rgb_mean_std` / `rgb_histogram` for benchmarks. Lower strength for histogram matching.

The `strength=0` run is a useful **no-color-match control**. Save both the full output and a zoom on the mask boundary; judge color continuity, texture, skin lighting and identity.

## Technical details and limitations

- After diffusion and VAE decode, the original crop is resized if needed, LAB/RGB matching applied, and the corrected crop returned. Impact Pack then downscales it and composites it with its own mask.
- All modes apply a global color transformation to the regenerated crop; **boundary-aware describes where statistics are sampled**, not a spatially varying seamless-clone algorithm.
- Correction may affect an entire regenerated crop internally; **only Impact Pack's final SEGS mask controls visible compositing**.
- To evaluate the real visible effect, compare the final composited `IMAGE` output rather than just the cropped/decoded preview.
- Small/empty masks are left unchanged.
- Supports `DetailerHookCombine`. If another hook adjusts decoded colors, place this color-match hook second to run its `post_decode` adjustment afterward.
- Verified with unit and mocked hook-interface tests. A real ComfyUI/RunPod end-to-end run is still required.

## Run tests

```bash
cd /opt/ComfyUI/custom_nodes/ComfyUI-SEGS-Detailer-Hook
python -m unittest discover -s tests -v
```

Tests cover all matching methods, backwards compatibility, mask behavior, and the Impact callback interface.


## SEGS Debug Overlay (new)

Search for **SEGS Debug Overlay** under `ImpactPack/Detailer/Hooks`. This standalone diagnostic node takes an `IMAGE` plus Impact Pack `SEGS`, and overlays detection details on the **original image**.

```text
Original image -------------------> SEGS Debug Overlay ----> Preview Image
                                            |                  (annotated)
BBOX Detector (SEGS) ----------------------> SEGS
                                            |
                                            +----> SEGS passthrough to Detailer (SEGS)
                                            +----> combined_segs_mask (MASK)
                                            +----> combined_matching_mask (MASK)
```

The overlay uses color-coded layers:

| Color | Visualization | Widget |
| --- | --- | --- |
| Amber | The original detected bounding box, with confidence/label | `show_bbox` |
| Cyan | The crop region after `bbox_crop_factor` and any SEGS transformations | `show_crop_region` |
| Green | The SEGS mask actually supplied to the detailer | `show_segs_mask` |
| Magenta | Pixels used by the selected color-matching method to calculate statistics | `show_matching_region` |
| Blue | Optional full-image include-mask preview | `show_include_mask` |
| Red | Optional full-image exclude-mask preview | `show_exclude_mask` |

Adjust `overlay_alpha` and `matching_alpha` independently from 0 to 1. Boxes and crop outlines remain visible even when masks are hidden. `thickness=0` automatically sizes lines for the image.

The overlay also outputs:

- `image`: annotated RGB image for Preview Image or Save Image.
- `segs`: original unchanged SEGS for the downstream Detailer (SEGS).
- `combined_segs_mask`: the full-image union of detected region masks.
- `combined_matching_mask`: the full-image union of pixels selected for color statistics. These two mask outputs do **not** depend on which overlays are enabled.

### Matching-region preview

Set `method`, `reference_region`, `mask_erosion`, and `boundary_width` to the same values as your **Color Match Detailer Hook (Multi-Mode)** node. For `boundary_chroma`, it visualizes the *inner mask boundary* used for color statistics, not a separate face-detection mask.

You can connect `include_mask_opt` and `exclude_mask_opt` to inspect how semantic segmentation would constrain the sample. **These inputs currently affect only the debug overlay, not the Color Match Detailer Hook**. The hook does not yet expose external include/exclude mask inputs; until that is added, these are exploratory visualizations.

**Notes and limits:**
- Matching-region coordinates are calculated at the **input image/crop resolution**. Impact Pack may upscale the crop internally before color matching; if so, `mask_erosion` and `boundary_width` may not correspond to identical numbers of original-image pixels.
- The preview assumes the detailer's `noise_mask` is enabled. If it is disabled, the hook will fall back to whole-crop statistics despite a SEGS mask being present.
- The overlay uses the original RGB pixels for region selection; its tints never feed into the matching calculations.
- The BBOX and crop outlines show the current SEGS object, so they reflect any prior SEGS transformations. A crop may be clamped at the image edge.
- For the most reliable diagnostics, use the same source resolution as the detector and inspect `combined_matching_mask` with Preview Mask.


## Mask logic expressions — diffusion masks (new)

Three additional nodes let you combine multiple full-image masks, then use the result as an Impact Pack **Detailer (SEGS)** diffusion/compositing mask. They do **not** generate semantic masks themselves: connect mask producers for skin, head, hair, eyes, etc.

### 1. SEGS BBox Mask

Inputs: original `IMAGE` and detector `SEGS`. Outputs `bbox_mask` (union of all detected bounding boxes in full-image coordinates) and unchanged `segs`.

### 2. Mask Logic Expression

Named optional MASK sockets: `bbox`, `human`, `skin`, `hair`, `head`, `face`, `eyes`, `nose`, `mouth`, `lips`, `ears`, `neck`, and `mask_a` through `mask_d`. Use `image_opt` to establish the source image dimensions. Unconnected names referenced in the expression give a readable error.

Expressions allow case-insensitive `AND`, `OR`, `XOR`, `NOT` or symbols `&`, `|`, `^`, `~`, plus parentheses. Boolean logic operates after thresholding each mask at `threshold` (default 0.50). Invalid Python syntax and function calls are rejected; this is **not** a Python `eval()` node.

Examples:

```text
bbox & skin
head & ~hair
(bbox & head) ^ hair
(face | head) & ~(eyes | lips)
```

Outputs: composed full-image `mask` and `resolved_expression` for logging.

### 3. Apply Mask To SEGS

Inputs: original `image`, original `segs`, composed full-image `mask`, `mode`, `threshold`, and `fallback_to_original_if_empty`.

- `replace`: overwrite each SEG's `cropped_mask` using the composed mask, cropped into that SEG's existing crop region.
- `and_existing`: intersect with the original SEG cropped mask.
- `or_existing`: union with the original mask within each crop.
- `xor_existing`: symmetric difference with the original mask.
- `fallback_to_original_if_empty`: disabled by default so an empty logical selection stays empty. Enable only if you intentionally want the old mask restored for empty results.

The original SEG's bbox, confidence, label, and `crop_region` remain unchanged. Because the crop does **not** expand automatically, portions of a composed mask falling outside it are clipped; enlarge the crop upstream if needed.

Outputs: updated `segs` and `combined_applied_mask` for Preview Mask.

### Suggested wiring

```text
BBOX/SEGS Detector -----------------------------+
  |                                              |
  v                                              v
SEGS BBox Mask -> bbox -------------------> Apply Mask To SEGS ---> Detailer (SEGS)
                                          ^                            |
Skin / Hair / Head semantic MASKs          |                            v
      |                                   |                     Color Match Hook
      +-> Mask Logic Expression ----------+
          expression: bbox & skin & ~hair
```

Set `noise_mask=enabled` in **Detailer (SEGS)** when you want the new SEGS mask applied during sampling. Temporarily use `noise_mask_feather=0` to inspect strict mask behavior. Impact Pack still uses the SEGS mask for the final composite.

You can send the updated SEGS to **SEGS Debug Overlay** to inspect the actual diffusion/compositing region. The color match hook receives this updated cropped mask when `noise_mask=enabled`, so using `reference_region=masked` provides a practical shared-mask workflow. **Independent full-image masks for color-matching statistics are not yet connected to the hook**, so do not expect the debug overlay's optional include/exclude masks to alter matching.

**Batch note:** The mask-expression node can process mask batches, but the SEGS override node intentionally requires one source image at a time, consistent with Impact Pack Detailer (SEGS). For multiple distinct faces, each SEG retains its own crop, and the mask is cropped independently for each one; overlapping crop regions may contain pixels from more than one composed-mask component.

### Test

```bash
cd /opt/ComfyUI/custom_nodes/ComfyUI-SEGS-Detailer-Hook
python -m unittest discover -s tests -v
```
