"""
Download the official llama.cpp build that matches the installed llama-cpp-python, verify it, and put it
where decidekv looks for it (~/.decidekv/). Needed when the llama-cpp-python wheel was compiled for CPU
instructions this machine lacks (crash 0xc000001d / "illegal instruction", e.g. no AVX-512).

    python -m decidekv.get_llamacpp
"""
from __future__ import annotations

import hashlib
import os
import sys
import urllib.request
import zipfile

# llama-cpp-python version -> (llama.cpp release, {asset: sha256}) ; official ggml-org/llama.cpp releases
BUILDS = {
    "0.3.36": ("b11312", {
        "llama-b11312-bin-win-cuda-12.4-x64.zip": "c7e67ed12791e31e21aba6a15856bade791586e4a1cefa313ea207f441089133",
        "cudart-llama-bin-win-cuda-12.4-x64.zip": "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6",
    }),
}
HOME = os.path.join(os.path.expanduser("~"), ".decidekv")


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if sys.platform != "win32":
        print("Only needed on Windows; on Linux/macOS build llama-cpp-python with CMAKE_ARGS for your GPU.")
        return
    try:
        from importlib.metadata import version
        ver = version("llama-cpp-python")
    except Exception:
        sys.exit("llama-cpp-python is not installed: pip install llama-cpp-python==0.3.36 "
                 "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu125")
    if ver not in BUILDS:
        sys.exit(f"no verified llama.cpp build registered for llama-cpp-python {ver}; known: {list(BUILDS)}")
    tag, assets = BUILDS[ver]
    dest = os.path.join(HOME, f"llama.cpp-{tag}")
    os.makedirs(dest, exist_ok=True)
    for name, digest in assets.items():
        path = os.path.join(dest, name)
        if not (os.path.exists(path) and sha256(path) == digest):
            url = f"https://github.com/ggml-org/llama.cpp/releases/download/{tag}/{name}"
            print("downloading", url)
            urllib.request.urlretrieve(url, path)
        got = sha256(path)
        if got != digest:
            os.remove(path)
            sys.exit(f"checksum mismatch for {name}: {got} != {digest} (file removed)")
        print("verified", name)
        with zipfile.ZipFile(path) as z:
            z.extractall(os.path.join(dest, "bin"))
    print("ready:", os.path.join(dest, "bin"), "(found automatically by decidekv)")


if __name__ == "__main__":
    main()
