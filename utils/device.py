"""Device selection utilities."""

import torch
from omegaconf import DictConfig


def resolve_device(cfg: DictConfig) -> torch.device:
    """
    Resolve the torch device based on Hydra configuration.

    Supports selecting a specific CUDA device via `cfg.cuda_device_index`.
    Falls back to CPU when the requested accelerator is unavailable.
    """
    target_device = str(cfg.device).lower()

    if target_device == "cuda":
        if not torch.cuda.is_available():
            print("CUDA requested but not available. Falling back to CPU.")
            return torch.device("cpu")

        index = int(getattr(cfg, "cuda_device_index", 0))
        if index < 0 or index >= torch.cuda.device_count():
            raise ValueError(
                f"Invalid cuda_device_index={index}. Available device count: {torch.cuda.device_count()}."
            )

        torch.cuda.set_device(index)
        return torch.device(f"cuda:{index}")

    if target_device == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")

        print("MPS requested but not available. Falling back to CPU.")
        return torch.device("cpu")

    return torch.device("cpu")
