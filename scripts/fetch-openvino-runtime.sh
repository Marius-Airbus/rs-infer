#!/usr/bin/env sh
# Fetch the shared libraries an `ep-openvino` build of rsinfer-server loads at
# startup: ONNX Runtime with its OpenVINO execution provider, plus OpenVINO and
# TBB, all bundled in Intel's onnxruntime-openvino wheel (Linux x86_64).
#
#   scripts/fetch-openvino-runtime.sh [DEST] [VERSION]
#
# Then run the server with ORT_DYLIB_PATH=$DEST/libonnxruntime.so.$VERSION.
# Needs python3 with pip (only to download and unpack the wheel).
set -eu
DEST=${1:-/opt/onnxruntime-openvino}
VERSION=${2:-1.24.1}
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

python3 -m pip download --quiet --no-deps --only-binary=:all: \
	--python-version 3.12 --platform manylinux_2_28_x86_64 \
	"onnxruntime-openvino==$VERSION" -d "$TMP"
mkdir -p "$DEST"
python3 - "$TMP"/*.whl "$DEST" <<'PY'
import os, sys, zipfile
wheel, dest = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(wheel) as z:
    for name in z.namelist():
        base = os.path.basename(name)
        # Shared libraries only; the Python binding is not needed.
        if name.startswith("onnxruntime/capi/") and ".so" in base and "pybind" not in base:
            with z.open(name) as src, open(os.path.join(dest, base), "wb") as out:
                out.write(src.read())
PY
echo "ORT_DYLIB_PATH=$DEST/libonnxruntime.so.$VERSION"
