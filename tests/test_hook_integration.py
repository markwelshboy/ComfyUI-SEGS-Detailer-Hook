"""Smoke-test the Impact hook interface using lightweight mocks (no ComfyUI needed)."""
import importlib.util
import sys
import types
import unittest
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]


class TestHookInterface(unittest.TestCase):
    def test_capture_and_match_multiple_regions(self):
        # Simulate the interface exposed by Impact Pack without importing ComfyUI.
        hooks_mod = types.ModuleType('impact.hooks')
        hooks_mod.DetailerHook = type('DetailerHook', (), {})
        impact_mod = types.ModuleType('impact')
        impact_mod.hooks = hooks_mod
        pkg_name = 'segs_detailer_hook_smoketest'
        spec = importlib.util.spec_from_file_location(pkg_name, ROOT / '__init__.py', submodule_search_locations=[str(ROOT)])
        module = importlib.util.module_from_spec(spec)
        saved = {name: sys.modules.get(name) for name in ('impact', 'impact.hooks', pkg_name)}
        try:
            sys.modules['impact'] = impact_mod
            sys.modules['impact.hooks'] = hooks_mod
            sys.modules[pkg_name] = module
            spec.loader.exec_module(module)
            provider = module.ColorMatchDetailerHookProvider()
            self.assertIn('method', provider.INPUT_TYPES()['required'])
            for method in module.METHODS:
                hook, = provider.doit(method=method, strength=0.5)
                ref = torch.ones((1, 48, 48, 3), dtype=torch.float32) * torch.tensor([0.7, 0.5, 0.4])
                pred = torch.ones((1, 48, 48, 3), dtype=torch.float32) * torch.tensor([0.6, 0.5, 0.5])
                mask = torch.ones((1, 48, 48), dtype=torch.float32)
                hook.post_upscale(ref, mask)
                out = hook.post_decode(pred)
                self.assertEqual(tuple(out.shape), tuple(pred.shape))
                self.assertIsNone(hook._original_crop)
                self.assertFalse(torch.isnan(out).any())
                # Re-use same instance for next SEG; no stale references.
                hook.post_upscale(ref, mask)
                out = hook.post_decode(pred)
                self.assertEqual(tuple(out.shape), tuple(pred.shape))
        finally:
            for name, old in saved.items():
                if old is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = old


if __name__ == '__main__':
    unittest.main()
