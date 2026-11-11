"""Boolean mask expressions and SEGS mask override for Impact Pack.

All expressions operate on aligned full-image binary masks; the resulting
mask is cropped to each existing SEG.crop_region without changing the crop.
No eval/exec is used: only a restricted Python AST is interpreted.
"""
import ast
import copy
import re

import cv2
import numpy as np
import torch

MASK_NAMES = ("bbox", "human", "skin", "hair", "head", "face", "eyes",
              "nose", "mouth", "lips", "ears", "neck",
              "mask_a", "mask_b", "mask_c", "mask_d")


def _array(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _as_batch(value, batch, height, width):
    arr = _array(value).astype(np.float32)
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim == 2:
        arr = arr[None]
    if arr.ndim != 3:
        raise ValueError(f"Expected MASK HxW or BxHxW, got {arr.shape}")
    if arr.shape[0] == 1 and batch != 1:
        arr = np.repeat(arr, batch, axis=0)
    if arr.shape[0] != batch:
        raise ValueError(f"Mask batch {arr.shape[0]} does not match image batch {batch}")
    if arr.shape[1:] != (height, width):
        arr = np.stack([cv2.resize(a, (width, height),
                                   interpolation=cv2.INTER_LINEAR) for a in arr])
    return np.clip(arr, 0.0, 1.0)


def _normalize(expr):
    names = {"and": "&", "or": "|", "xor": "^", "not": "~"}
    return re.sub(r"\b(?:AND|OR|XOR|NOT)\b",
                  lambda m: names[m.group(0).lower()], expr.strip(),
                  flags=re.IGNORECASE)


def _parse(expr, env):
    class SafeEval(ast.NodeVisitor):
        def visit_Expression(self, node):
            return self.visit(node.body)

        def visit_Name(self, node):
            if node.id not in env:
                raise ValueError(f"Mask '{node.id}' is not connected")
            return env[node.id]

        def visit_UnaryOp(self, node):
            if not isinstance(node.op, ast.Invert):
                raise ValueError("Only NOT/~ is supported as a unary operator")
            return ~self.visit(node.operand)

        def visit_BinOp(self, node):
            left, right = self.visit(node.left), self.visit(node.right)
            if isinstance(node.op, ast.BitAnd):
                return left & right
            if isinstance(node.op, ast.BitOr):
                return left | right
            if isinstance(node.op, ast.BitXor):
                return left ^ right
            raise ValueError("Supported operators are AND, OR, XOR and NOT")

        def generic_visit(self, node):
            raise ValueError(f"Unsafe/unsupported expression syntax: {type(node).__name__}")

    return SafeEval().visit(ast.parse(_normalize(expr), mode="eval"))


def _crop(mask, rect):
    x1, y1, x2, y2 = map(int, rect)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid crop region: {rect}")
    h, w = mask.shape
    out = np.zeros((y2-y1, x2-x1), np.float32)
    a, b, c, d = max(0,x1), max(0,y1), min(w,x2), min(h,y2)
    if a < c and b < d:
        out[b-y1:d-y1, a-x1:c-x1] = mask[b:d, a:c]
    return out


def _paste_union(full, crop_mask, rect):
    x1, y1, x2, y2 = map(int, rect)
    h, w = full.shape
    a, b, c, d = max(0,x1), max(0,y1), min(w,x2), min(h,y2)
    if a < c and b < d:
        full[b:d, a:c] = np.maximum(full[b:d, a:c],
                                    crop_mask[b-y1:d-y1, a-x1:c-x1])


def _replace_seg(seg, new_mask):
    if hasattr(seg, "_replace"):
        return seg._replace(cropped_mask=new_mask)
    clone = copy.copy(seg)
    try:
        clone.cropped_mask = new_mask
    except Exception as exc:
        raise TypeError("Unsupported SEG type; expected Impact Pack namedtuple") from exc
    return clone


class SegsBBoxMaskNode:
    CATEGORY = "ImpactPack/Detailer/Hooks"
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "segs": ("SEGS",)}}
    RETURN_TYPES = ("MASK", "SEGS")
    RETURN_NAMES = ("bbox_mask", "segs")
    FUNCTION = "make_mask"

    def make_mask(self, image, segs):
        batch, h, w, _ = image.shape
        (sh, sw), segments = segs
        union = np.zeros((h, w), np.float32)
        for seg in segments:
            x1, y1, x2, y2 = seg.bbox
            x1, x2 = int(np.floor(x1*w/sw)), int(np.ceil(x2*w/sw))
            y1, y2 = int(np.floor(y1*h/sh)), int(np.ceil(y2*h/sh))
            union[max(0,y1):min(h,y2), max(0,x1):min(w,x2)] = 1.0
        return (torch.from_numpy(np.repeat(union[None], batch, axis=0)), segs)


class MaskExpressionNode:
    CATEGORY = "ImpactPack/Detailer/Hooks"
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "expression": ("STRING", {"default": "bbox & skin", "multiline": False}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
            "optional": {
                **{name: ("MASK",) for name in MASK_NAMES},
                "image_opt": ("IMAGE",),
            },
        }
    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("mask", "resolved_expression")
    FUNCTION = "evaluate"

    def evaluate(self, expression, threshold=0.5, image_opt=None, **inputs):
        present = {name: mask for name, mask in inputs.items()
                   if name in MASK_NAMES and mask is not None}
        if not present:
            raise ValueError("Connect at least one named mask")
        if image_opt is not None:
            batch, h, w = map(int, image_opt.shape[:3])
        else:
            first = _array(next(iter(present.values())))
            if first.ndim == 4 and first.shape[-1] == 1:
                first = first[..., 0]
            batch, h, w = (1, *first.shape) if first.ndim == 2 else first.shape[:3]
        env = {name: _as_batch(mask, batch, h, w) >= threshold
               for name, mask in present.items()}
        result = _parse(expression, env)
        return (torch.from_numpy(result.astype(np.float32)), _normalize(expression))


class ApplyMaskToSegsNode:
    CATEGORY = "ImpactPack/Detailer/Hooks"
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "segs": ("SEGS",),
            "mask": ("MASK",),
            "mode": (["replace", "and_existing", "or_existing", "xor_existing"],),
            "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
            "fallback_to_original_if_empty": ("BOOLEAN", {"default": False}),
        }}
    RETURN_TYPES = ("SEGS", "MASK")
    RETURN_NAMES = ("segs", "combined_applied_mask")
    FUNCTION = "apply"

    def apply(self, image, segs, mask, mode="replace", threshold=0.5,
              fallback_to_original_if_empty=False):
        batch, h, w = map(int, image.shape[:3])
        if batch != 1:
            raise ValueError("Impact Pack Detailer (SEGS) is single-image; pass one image at a time")
        (sh, sw), segments = segs
        aligned = _as_batch(mask, 1, sh, sw)[0] >= threshold
        combined = np.zeros((sh, sw), np.float32)
        updated = []
        for seg in segments:
            rect = tuple(map(int, seg.crop_region))
            chosen = _crop(aligned.astype(np.float32), rect) >= 0.5
            existing = _array(seg.cropped_mask).astype(np.float32)
            if existing.ndim == 3:
                existing = existing[0]
            if existing.shape != chosen.shape:
                existing = cv2.resize(existing, (chosen.shape[1], chosen.shape[0]),
                                      interpolation=cv2.INTER_LINEAR)
            old = existing >= threshold
            if mode == "replace":
                result = chosen
            elif mode == "and_existing":
                result = old & chosen
            elif mode == "or_existing":
                result = old | chosen
            elif mode == "xor_existing":
                result = old ^ chosen
            else:
                raise ValueError(f"Unknown mode {mode}")
            if not result.any() and fallback_to_original_if_empty:
                result = old
            result_mask = result.astype(np.float32)
            updated.append(_replace_seg(seg, result_mask))
            _paste_union(combined, result_mask, rect)
        # Each SEG keeps its original bbox and crop_region. Masks outside the crop
        # are intentionally clipped; the final SEGS mask drives composition.
        return (((sh, sw), updated), torch.from_numpy(combined[None]))
