"""Compare native filtered GGUF views against their real component checkpoints.

Only no_alloc metadata contexts are created; no accelerator is initialized.
Every selected tensor's original offset, type, shape, size and payload bytes,
and all original metadata including tokenizer arrays, must remain identical.
"""
from __future__ import annotations

import argparse
import ctypes as c
import json
from pathlib import Path


class Params(c.Structure):
    _fields_ = [("no_alloc", c.c_bool), ("ctx", c.POINTER(c.c_void_p))]


class Native:
    def __init__(self, library: Path):
        self.library = c.CDLL(str(library.resolve(strict=True)))
        signatures = {
            "gguf_init_from_file": (c.c_void_p, [c.c_char_p, Params]),
            "gguf_free": (None, [c.c_void_p]), "ggml_free": (None, [c.c_void_p]),
            "gguf_select_tensor_view": (c.c_bool, [c.c_void_p, c.c_char_p, c.c_char_p, c.POINTER(c.c_void_p)]),
            "gguf_get_n_kv": (c.c_int64, [c.c_void_p]),
            "gguf_get_n_tensors": (c.c_int64, [c.c_void_p]),
            "gguf_get_data_offset": (c.c_size_t, [c.c_void_p]),
            "gguf_get_alignment": (c.c_size_t, [c.c_void_p]),
            "gguf_get_key": (c.c_char_p, [c.c_void_p, c.c_int64]),
            "gguf_get_tensor_name": (c.c_char_p, [c.c_void_p, c.c_int64]),
            "gguf_get_tensor_type": (c.c_int, [c.c_void_p, c.c_int64]),
            "gguf_get_tensor_size": (c.c_size_t, [c.c_void_p, c.c_int64]),
            "gguf_get_tensor_offset": (c.c_size_t, [c.c_void_p, c.c_int64]),
            "gguf_get_tensor_ne": (c.POINTER(c.c_int64), [c.c_void_p, c.c_int64]),
            "gguf_get_kv_type": (c.c_int, [c.c_void_p, c.c_int64]),
            "gguf_get_arr_type": (c.c_int, [c.c_void_p, c.c_int64]),
            "gguf_get_arr_n": (c.c_size_t, [c.c_void_p, c.c_int64]),
            "gguf_get_arr_str": (c.c_char_p, [c.c_void_p, c.c_int64, c.c_size_t]),
            "gguf_get_arr_data": (c.c_void_p, [c.c_void_p, c.c_int64]),
            "gguf_get_val_str": (c.c_char_p, [c.c_void_p, c.c_int64]),
            "gguf_get_val_data": (c.c_void_p, [c.c_void_p, c.c_int64]),
            "ggml_get_tensor": (c.c_void_p, [c.c_void_p, c.c_char_p]),
            "ggml_nbytes": (c.c_size_t, [c.c_void_p]),
        }
        for name, (result, args) in signatures.items():
            function = getattr(self.library, name)
            function.restype, function.argtypes = result, args
            setattr(self, name.removeprefix("gguf_"), function)

    def metadata(self, ctx, index):
        kind = self.get_kv_type(ctx, index)
        widths = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        if kind == 8:
            return kind, self.get_val_str(ctx, index)
        if kind == 9:
            subtype, count = self.get_arr_type(ctx, index), self.get_arr_n(ctx, index)
            values = (tuple(self.get_arr_str(ctx, index, n) for n in range(count)) if subtype == 8
                      else c.string_at(self.get_arr_data(ctx, index), count * widths[subtype]))
            return kind, subtype, count, values
        return kind, c.string_at(self.get_val_data(ctx, index), widths[kind])


def verify(native: Native, bundle: Path, source: Path, name: str, prefix: bytes) -> dict:
    view_meta, source_meta = c.c_void_p(), c.c_void_p()
    view = native.init_from_file(str(bundle).encode(), Params(True, c.pointer(view_meta)))
    original = native.init_from_file(str(source).encode(), Params(True, c.pointer(source_meta)))
    assert view and original
    try:
        tensor_count, key_count, pointer = native.get_n_tensors(view), native.get_n_kv(view), view_meta.value
        physical_start, alignment = native.get_data_offset(view), native.get_alignment(view)
        offsets = {native.get_tensor_name(view, n): native.get_tensor_offset(view, n)
                   for n in range(tensor_count)}
        metadata_prefix = f"robit.audio_bundle.component.{name}.kv.".encode()
        for bad_tensor, bad_meta in ((b"missing.", metadata_prefix), (prefix, b"missing."), (b"", metadata_prefix)):
            assert not native.select_tensor_view(view, bad_tensor, bad_meta, c.pointer(view_meta))
            assert (native.get_n_tensors(view), native.get_n_kv(view), view_meta.value) == \
                (tensor_count, key_count, pointer), "failed selection modified the source context"
        assert native.select_tensor_view(view, prefix, metadata_prefix, c.pointer(view_meta))
        assert native.get_data_offset(view) == physical_start and native.get_alignment(view) == alignment
        assert native.get_n_kv(view) == native.get_n_kv(original)
        for n in range(native.get_n_kv(original)):
            assert native.get_key(view, n) == native.get_key(original, n)
            assert native.metadata(view, n) == native.metadata(original, n)
        assert native.get_n_tensors(view) == native.get_n_tensors(original)
        payload = 0
        types = set()
        with bundle.open("rb") as packed, source.open("rb") as standalone:
            for n in range(native.get_n_tensors(original)):
                tensor_name = native.get_tensor_name(original, n)
                assert native.get_tensor_name(view, n) == tensor_name
                assert native.get_tensor_type(view, n) == native.get_tensor_type(original, n)
                assert list(native.get_tensor_ne(view, n)[:4]) == list(native.get_tensor_ne(original, n)[:4])
                size = native.get_tensor_size(original, n)
                assert native.get_tensor_size(view, n) == size
                assert native.get_tensor_offset(view, n) == offsets[prefix + tensor_name]
                tensor = native.ggml_get_tensor(view_meta, tensor_name)
                assert tensor and native.ggml_nbytes(tensor) == size
                types.add(native.get_tensor_type(view, n))
                packed.seek(physical_start + native.get_tensor_offset(view, n))
                standalone.seek(native.get_data_offset(original) + native.get_tensor_offset(original, n))
                remaining = size
                while remaining:
                    count = min(remaining, 1024 * 1024)
                    left, right = packed.read(count), standalone.read(count)
                    assert len(left) == len(right) == count and left == right, tensor_name
                    remaining -= count
                payload += size
        return {"component": name, "tensor_count": native.get_n_tensors(view),
                "metadata_keys": native.get_n_kv(view), "tensor_types": sorted(types),
                "payload_bytes_compared": payload, "original_offsets_preserved": True,
                "all_metadata_shapes_types_and_payloads_identical": True,
                "failed_selection_preserves_original_context": True}
    finally:
        native.ggml_free(view_meta)
        native.ggml_free(source_meta)
        native.free(view)
        native.free(original)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("library", "bundle", "model", "projector", "out"):
        parser.add_argument("--" + argument, type=Path, required=True)
    args = parser.parse_args()
    native = Native(args.library)
    result = {"schema": "robit.omni.native-view-verification.v1", "components": [
        verify(native, args.bundle, args.model, "tts_model", b"s.t.m."),
        verify(native, args.bundle, args.projector, "tts_projector", b"s.t.p.")]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
