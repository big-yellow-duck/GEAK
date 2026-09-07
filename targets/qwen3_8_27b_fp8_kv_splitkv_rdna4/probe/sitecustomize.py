"""Install a non-invasive import hook for paged-attention call capture."""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import json
import os
import sys
from types import ModuleType
from typing import Any

_MODULE = "vllm.v1.attention.ops.chunked_prefill_paged_decode"
_ENV = "GEAK_SPLITKV_CAPTURE_LOG"


def _tensor_meta(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    shape = getattr(value, "shape", None)
    stride = getattr(value, "stride", None)
    if shape is None or not callable(stride):
        return None
    return {
        "shape": [int(dim) for dim in shape],
        "stride": [int(dim) for dim in stride()],
        "dtype": str(getattr(value, "dtype", "")),
        "device": str(getattr(value, "device", "")),
    }


def _scalar_value(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if int(value.numel()) != 1:
            return None
        return float(value.detach().item())
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None


def _append_record(path: str, record: dict[str, Any]) -> None:
    payload = (json.dumps(record, sort_keys=True) + "\n").encode()
    fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(fd, payload)
    finally:
        os.close(fd)


def _install_wrapper(module: ModuleType) -> None:
    original = module.chunked_prefill_paged_decode
    if getattr(original, "_geak_splitkv_capture", False):
        return

    def wrapped(
        query,
        key,
        value,
        output,
        kv_cache_dtype,
        key_cache,
        value_cache,
        block_table,
        query_start_loc,
        seq_lens,
        max_seq_len,
        max_query_len,
        k_scale,
        v_scale,
        *args,
        **kwargs,
    ):
        path = os.environ.get(_ENV)
        if path:
            _append_record(
                path,
                {
                    "schema": "geak.splitkv_paged_decode_call.v1",
                    "pid": os.getpid(),
                    "rank": os.environ.get("RANK"),
                    "local_rank": os.environ.get("LOCAL_RANK"),
                    "kv_cache_dtype": str(kv_cache_dtype),
                    "max_seq_len": int(max_seq_len),
                    "max_query_len": int(max_query_len),
                    "query": _tensor_meta(query),
                    "key": _tensor_meta(key),
                    "value": _tensor_meta(value),
                    "output": _tensor_meta(output),
                    "key_cache": _tensor_meta(key_cache),
                    "value_cache": _tensor_meta(value_cache),
                    "block_table": _tensor_meta(block_table),
                    "query_start_loc": _tensor_meta(query_start_loc),
                    "seq_lens": _tensor_meta(seq_lens),
                    "k_scale": _tensor_meta(k_scale),
                    "v_scale": _tensor_meta(v_scale),
                    "k_scale_value": _scalar_value(k_scale),
                    "v_scale_value": _scalar_value(v_scale),
                    "output_scale": _tensor_meta(kwargs.get("output_scale")),
                    "is_block_table_ptr": bool(kwargs.get("is_block_table_ptr", False)),
                    "causal": bool(kwargs.get("causal", True)),
                    "unit_kv_scale": bool(kwargs.get("unit_kv_scale", False)),
                },
            )
        return original(
            query,
            key,
            value,
            output,
            kv_cache_dtype,
            key_cache,
            value_cache,
            block_table,
            query_start_loc,
            seq_lens,
            max_seq_len,
            max_query_len,
            k_scale,
            v_scale,
            *args,
            **kwargs,
        )

    wrapped._geak_splitkv_capture = True
    module.chunked_prefill_paged_decode = wrapped


class _Loader(importlib.abc.Loader):
    def __init__(self, wrapped: importlib.abc.Loader) -> None:
        self.wrapped = wrapped

    def create_module(self, spec):
        create = getattr(self.wrapped, "create_module", None)
        return create(spec) if create is not None else None

    def exec_module(self, module: ModuleType) -> None:
        self.wrapped.exec_module(module)
        _install_wrapper(module)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path, target=None):
        if fullname != _MODULE:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return spec
        spec.loader = _Loader(spec.loader)
        return spec


sys.meta_path.insert(0, _Finder())
