# syntax=docker/dockerfile:1
# GPU EP combos (pyke prebuilt binaries, CUDA/TensorRT exist only for linux/amd64):
#   ep-cuda | ep-cuda,ep-tensorrt | ep-nvrtx,lax-ep-matching
# To build the GPU image from Apple Silicon, run: docker build --platform linux/amd64 .
# Leave FEATURES unset for target-aware auto-selection (amd64 -> GPU EPs, other -> CPU-only).
# ONNX Runtime prebuilts need libstdc++ >= GCC 13 (trixie), not bookworm's GCC 12.
# Intel OpenVINO (experimental, linux/amd64):
#   docker build --build-arg FEATURES=ep-openvino --target runtime-openvino .
FROM rust:1.96-trixie AS build
ARG TARGETPLATFORM
ARG FEATURES
WORKDIR /src
COPY . .
RUN set -eux; \
    if [ -z "${FEATURES:-}" ]; then \
        case "${TARGETPLATFORM:-linux/amd64}" in \
            linux/amd64) FEATURES="ep-cuda,ep-tensorrt" ;; \
            *) FEATURES="" ;; \
        esac; \
    fi; \
    case ",$FEATURES," in \
        *,ep-openvino,*) cargo build --release -p rsinfer-server --no-default-features --features "$FEATURES" ;; \
        ,,) cargo build --release -p rsinfer-server ;; \
        *) cargo build --release -p rsinfer-server --features "$FEATURES" ;; \
    esac

# ONNX Runtime + OpenVINO shared libraries, for the runtime-openvino target only.
FROM python:3.12-slim AS openvino-libs
COPY scripts/fetch-openvino-runtime.sh /fetch-openvino-runtime.sh
RUN sh /fetch-openvino-runtime.sh /opt/onnxruntime-openvino 1.24.1

FROM debian:trixie-slim AS runtime-openvino
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates libgomp1 libstdc++6 curl && rm -rf /var/lib/apt/lists/*
COPY --from=openvino-libs /opt/onnxruntime-openvino /opt/onnxruntime-openvino
ENV ORT_DYLIB_PATH=/opt/onnxruntime-openvino/libonnxruntime.so.1.24.1 \
    LD_LIBRARY_PATH=/opt/onnxruntime-openvino
COPY --from=build /src/target/release/rsinfer-server /usr/local/bin/rsinfer-server
EXPOSE 8080
USER nobody
ENTRYPOINT ["/usr/local/bin/rsinfer-server"]
CMD ["--config", "/etc/rsinfer/config.yaml"]

# Default target: prebuilt ONNX Runtime linked into the binary.
FROM debian:trixie-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates libgomp1 libstdc++6 curl && rm -rf /var/lib/apt/lists/*
# Add CUDA/cuDNN/TensorRT runtime libraries to this image (or mount them) when using GPU EPs.
COPY --from=build /src/target/release/rsinfer-server /usr/local/bin/rsinfer-server
EXPOSE 8080
USER nobody
ENTRYPOINT ["/usr/local/bin/rsinfer-server"]
CMD ["--config", "/etc/rsinfer/config.yaml"]
