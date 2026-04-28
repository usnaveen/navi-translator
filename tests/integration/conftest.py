"""Stub heavy ML deps so CI can import the FastAPI app without installing
torch / transformers / librosa. The integration tests swap in a FakeEngine,
so the real inference code never runs — we just need the imports to succeed."""

import sys
import types


def _stub(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


for name in [
    "librosa",
    "torch",
    "torch.nn",
    "torch.nn.functional",
    "torchaudio",
    "transformers",
    "peft",
]:
    if name not in sys.modules:
        _stub(name)
