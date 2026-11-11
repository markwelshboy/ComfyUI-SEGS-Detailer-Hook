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
