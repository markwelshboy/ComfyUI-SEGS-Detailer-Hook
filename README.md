# ComfyUI SEGS Detailer Hook — LAB Color Match

A ComfyUI custom node that adds **mask-aware LAB color matching** to the [ComfyUI-Impact-Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack) **Detailer (SEGS)** and **FaceDetailer** nodes.

The hook captures the original detailer crop in `post_upscale()`, then adjusts the regenerated crop's color in `post_decode()` **after diffusion and VAE decoding but before the final masked composite**. No Impact Pack source modifications are needed.

## Installation

On RunPod or another ComfyUI host:

```bash
cd /opt/ComfyUI/custom_nodes
git clone https://github.com/markwelshboy/ComfyUI-SEGS-Detailer-Hook.git
```

Restart ComfyUI. Your ComfyUI installation needs Impact Pack, PyTorch, NumPy and OpenCV; these are normally already available in an installation that runs Impact Pack. The node imports Impact Pack's hook base at execution time, so startup import order does not matter.

### Updating

```bash
cd /opt/ComfyUI/custom_nodes/ComfyUI-SEGS-Detailer-Hook
git pull --ff-only
```

Restart ComfyUI after updating.

## Usage

Search for **Color Match Detailer Hook (LAB)** under `ImpactPack/Detailer/Hooks`.

Connect its `detailer_hook` output to the `detailer_hook` input on **Detailer (SEGS)** or **FaceDetailer**.

```text
BBOX Detector ───> SEGS ───> Detailer (SEGS) ───> Final image
                                 ▲
                                 │ detailer_hook
                      Color Match Detailer Hook (LAB)
```

The original crop and cropped noise mask are captured automatically; you do not need a separate image input. Each SEGS region is processed independently, using its **own original crop** for comparison rather than whole-image statistics.

## Inputs

| Setting | Default | What it does |
| --- | --- | --- |
| `enabled` | `true` | Enable or bypass matching |
| `strength` | `0.50` | Blend between uncorrected (0) and fully corrected (1) |
| `reference_region` | `masked` | Select masked pixels or the entire crop for color statistics |
| `match_luminance` | `false` | Leave generated luminance/contrast alone by default; match only LAB chroma |
| `mask_erosion` | `5` | Exclude a small number of pixels near mask edges when calculating statistics |

For Krea 2 facial detailing, try **strength 0.25–0.60, masked, match_luminance off, mask_erosion 5**.

**Masked mode needs `noise_mask = enabled`** on the detailer for the mask to reach the hook. If no mask is supplied, the hook falls back to whole-crop statistics. For diagnosing visible sampling transitions, try `noise_mask_feather = 0` separately; that is an Impact Pack diffusion setting, not a color-match setting.

## How the color correction works

1. Save the original crop at the working resolution and its optional noise mask.
2. After sampling and VAE decode, resize the original reference if needed to match the generated crop.
3. Calculate corresponding original and generated pixel statistics in CIELAB using either the masked interior or the whole crop.
4. Adjust chroma (LAB a/b) by default; optionally include lightness (L).
5. Interpolate the correction using `strength`, then return the crop to Impact Pack for its normal resizing and SEGS-mask compositing.

Using the masked region for statistics generally avoids background and clothing skewing facial colors. **A BBOX mask is rectangular and includes eyes, lips and possibly hair: the hook is mask-aware, not skin-specific.** For skin-only matching, provide a skin segmentation mask in a separate workflow.

## Combining hooks

Use Impact Pack's `DetailerHookCombine` node to combine this with compatible detailer hooks. Hooks are applied in order; if another hook modifies decoded colors, put this color-match hook **second** so its `post_decode()` adjustment runs after that change.

## Limitations

- Color matching is global across each crop; it doesn't perform spatially varying lighting correction or alter the mask boundary.
- The final affected pixels are controlled by Impact Pack's SEGS mask and `feather` setting, not by this hook.
- An empty or too-small masked selection (fewer than 64 pixels) is left unchanged.
- Regular ComfyUI RGB image crops and single-leading-dimension 5D outputs are handled; arbitrary video formats are not.
- The hook keeps temporary state while each SEGS region processes. It is intended for Impact Pack's sequential per-region processing.
- The Python unit tests cover color-statistics behavior but do not constitute an end-to-end ComfyUI integration test.

## Tests

From this repository directory, run:

```bash
python -m unittest discover -s tests -v
```

The tests require NumPy and OpenCV but not a running ComfyUI instance.
