"""Load a pickled PyTorch checkpoint only after torch's own scanner confirms every
referenced global comes from an allowlisted package (see inspect_pickle.py)."""
import torch

ALLOWED_PREFIXES = (
    "torch.",
    "timm.",
    "ultralytics.",
    "collections.",
    "numpy.",
    "__builtin__.set",
    "builtins.set",
    "_codecs.encode",
)


def safe_torch_load(path: str):
    unsafe = torch.serialization.get_unsafe_globals_in_checkpoint(path)
    bad = [g for g in unsafe if not g.startswith(ALLOWED_PREFIXES)]
    if bad:
        raise RuntimeError(f"{path} references non-allowlisted globals: {bad}")
    # Every global is a torch/timm/ultralytics class or plain container, so full
    # unpickling (needed to rebuild the nn.Module objects) cannot run foreign code.
    return torch.load(path, map_location="cpu", weights_only=False)
