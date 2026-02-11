"""Runtime patches for upstream sglang issues.

Currently we wrap the Triton token-pool kernel so that any metadata tensors
handed to it live on the same CUDA device as the token pool itself. The
upstream 0.3.x wheels occasionally construct those tensors on CPU, which causes
"illegal memory access" crashes as soon as the kernel dereferences the pointer.
This patch mirrors the upstream fix without requiring users to reinstall
sglang.
"""
from __future__ import annotations

import logging
from typing import Any

import torch

LOG = logging.getLogger(__name__)


def _maybe_to_device(tensor: Any, device: torch.device | None) -> Any:
    if not isinstance(tensor, torch.Tensor) or device is None:
        return tensor
    if tensor.device == device:
        return tensor
    if device.type == "cuda":
        return tensor.to(device, non_blocking=True)
    return tensor.to(device)


class _TokenPoolKernelWrapper:
    """Wrap Triton kernel launches and fix tensor devices on the fly."""

    def __init__(self, kernel):
        self._kernel = kernel

    def __getitem__(self, key):
        launch = self._kernel[key]

        def _wrapped(req_to_token, req_pool_indices, prefix_lens, seq_lens, extend_lens, out_cache_loc, pool_width):
            device = getattr(req_to_token, "device", None)
            req_pool_indices = _maybe_to_device(req_pool_indices, device)
            prefix_lens = _maybe_to_device(prefix_lens, device)
            seq_lens = _maybe_to_device(seq_lens, device)
            extend_lens = _maybe_to_device(extend_lens, device)
            out_cache_loc = _maybe_to_device(out_cache_loc, device)
            return launch(
                req_to_token,
                req_pool_indices,
                prefix_lens,
                seq_lens,
                extend_lens,
                out_cache_loc,
                pool_width,
            )

        return _wrapped

    def __getattr__(self, name):
        return getattr(self._kernel, name)

    def __call__(self, *args, **kwargs):
        return self._kernel(*args, **kwargs)


def apply_patches() -> None:
    """Apply all available runtime patches exactly once."""

    try:
        from sglang.srt.managers import schedule_batch as schedule_batch_mod
    except Exception:  # pragma: no cover - defensive import guard
        LOG.warning("Failed to import sglang.srt.managers.schedule_batch for patching", exc_info=True)
        return

    kernel = getattr(schedule_batch_mod, "write_req_to_token_pool_triton", None)
    if kernel is None:
        LOG.warning("write_req_to_token_pool_triton is missing; skip Triton patch")
        return

    if isinstance(kernel, _TokenPoolKernelWrapper):
        return

    schedule_batch_mod.write_req_to_token_pool_triton = _TokenPoolKernelWrapper(kernel)
    LOG.info("Applied Triton token-pool device patch (wrap launch kernel)")


# Auto-apply when imported so users only need to import this module once.
apply_patches()
