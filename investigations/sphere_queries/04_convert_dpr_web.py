"""Convert the Sphere DPR_web .cp pickle checkpoint to safetensors without torch.

Run only inside the isolated venv with a clean environment (see README). The pickle
is read through an allowlist unpickler: OrderedDict, torch tensor rebuild and storage
types are mapped to inert numpy-backed objects; every other global becomes a stub
that records its name and cannot execute anything.

usage: 04_convert_dpr_web.py CHECKPOINT OUT_SAFETENSORS REPORT_JSON
"""

import json
import pickle
import socket
import sys
import zipfile
from collections import OrderedDict
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

STORAGE_DTYPES = {
    "FloatStorage": np.float32,
    "HalfStorage": np.float16,
    "LongStorage": np.int64,
    "IntStorage": np.int32,
    "DoubleStorage": np.float64,
    "BoolStorage": np.bool_,
    "ByteStorage": np.uint8,
}
STUBBED: set[str] = set()


class Storage:
    def __init__(self, dtype, key):
        self.dtype = dtype
        self.key = key


class LazyTensor:
    def __init__(self, storage, offset, size, stride):
        self.storage = storage
        self.offset = offset
        self.size = tuple(size)
        self.stride = tuple(stride)


class Stub:
    def __init__(self, *args, **kwargs):
        self.args = args

    def __setstate__(self, state):
        self.state = state

    def __setitem__(self, key, value):
        self.__dict__.setdefault("items", {})[key] = value

    def append(self, value):
        self.__dict__.setdefault("seq", []).append(value)

    def extend(self, values):
        self.__dict__.setdefault("seq", []).extend(values)


def rebuild_tensor_v2(storage, offset, size, stride, requires_grad=False,
                      backward_hooks=None, metadata=None):
    return LazyTensor(storage, offset, size, stride)


class AllowlistUnpickler(pickle.Unpickler):
    def __init__(self, handle, archive: zipfile.ZipFile, prefix: str):
        super().__init__(handle)
        self.archive = archive
        self.prefix = prefix

    def find_class(self, module, name):
        if (module, name) == ("collections", "OrderedDict"):
            return OrderedDict
        if module == "torch" and name in STORAGE_DTYPES:
            return STORAGE_DTYPES[name]
        if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return rebuild_tensor_v2
        STUBBED.add(f"{module}.{name}")
        return type(name, (Stub,), {})

    def persistent_load(self, pid):
        tag, dtype, key = pid[0], pid[1], pid[2]
        if tag != "storage":
            raise pickle.UnpicklingError(f"unexpected persistent id {tag!r}")
        return Storage(dtype, key)


def materialize(tensor: LazyTensor, archive: zipfile.ZipFile, prefix: str) -> np.ndarray:
    raw = archive.read(f"{prefix}/data/{tensor.storage.key}")
    flat = np.frombuffer(raw, dtype=tensor.storage.dtype)
    itemsize = flat.itemsize
    view = np.lib.stride_tricks.as_strided(
        flat[tensor.offset:], shape=tensor.size,
        strides=tuple(s * itemsize for s in tensor.stride))
    return np.array(view)


def network_check() -> str:
    try:
        sock = socket.create_connection(("1.1.1.1", 443), timeout=5)
        sock.close()
        return "connected (network NOT blocked)"
    except OSError as err:
        return f"blocked: {err!r}"


def main() -> None:
    checkpoint, out_path, report_path = (Path(a) for a in sys.argv[1:4])
    report = {"network_check": network_check()}
    archive = zipfile.ZipFile(checkpoint)
    pkl_name = next(n for n in archive.namelist() if n.endswith("/data.pkl"))
    prefix = pkl_name.rsplit("/", 1)[0]
    with archive.open(pkl_name) as handle:
        state = AllowlistUnpickler(handle, archive, prefix).load()
    report["top_level_keys"] = sorted(state.keys()) if isinstance(state, dict) else str(type(state))
    model_dict = state["model_dict"]
    tensors = {}
    for name, value in model_dict.items():
        if isinstance(value, LazyTensor):
            tensors[name] = materialize(value, archive, prefix)
    report["n_model_tensors"] = len(tensors)
    report["stubbed_globals"] = sorted(STUBBED)
    report["shapes"] = {k: list(v.shape) for k, v in tensors.items()
                        if "embeddings" in k or "pooler" in k or "position_ids" in k}
    report["prefixes"] = sorted({k.split(".")[0] for k in tensors})
    save_file({k: np.ascontiguousarray(v) for k, v in tensors.items()}, str(out_path))
    report_path.write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps({k: report[k] for k in ("network_check", "top_level_keys", "n_model_tensors",
                                              "stubbed_globals", "prefixes")}, indent=1))


if __name__ == "__main__":
    main()
