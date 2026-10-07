# Build profiles (ONNX Runtime prebuilt binaries do not combine all EPs).
BIN = rsinfer-server

.PHONY: mac mac-coreml gpu-cuda gpu-trt gpu-rtx cpu-openvino cpu test fmt lint docs docs-build docs-clean

cpu:            ## CPU-only (works everywhere)
	cargo build --release -p rsinfer-server

mac-coreml:     ## macOS + CoreML/ANE
	cargo build --release -p $(BIN) --features ep-coreml

gpu-cuda:       ## Linux + CUDA
	cargo build --release -p $(BIN) --features ep-cuda

gpu-trt:        ## Linux + TensorRT (datacenter GPUs, includes CUDA)
	cargo build --release -p $(BIN) --features ep-cuda,ep-tensorrt

gpu-rtx:        ## Linux + TensorRT for RTX (consumer GeForce/RTX cards)
	cargo build --release -p $(BIN) --features ep-nvrtx,lax-ep-matching

cpu-openvino:   ## Linux + Intel OpenVINO (experimental): ONNX Runtime loaded at runtime, see scripts/fetch-openvino-runtime.sh
	cargo build --release -p $(BIN) --no-default-features --features ep-openvino

mac: mac-coreml
test:
	cargo test --workspace
fmt:
	cargo fmt --all
lint:
	cargo clippy --workspace --all-targets

docs:           ## Serve the docs site locally (http://localhost:3000)
	cd docs-site && npm install && npm run dev

docs-build:     ## Static-export the docs site to docs-site/out (GitHub Pages-ready)
	cd docs-site && npm install && npm run build

docs-clean:
	rm -rf docs-site/out docs-site/.next docs-site/.source
