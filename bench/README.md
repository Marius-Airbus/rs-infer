# Benchmarks

`run.sh` load-tests a running server with fixed scenarios, `compare.py` checks
that two servers return the same embeddings. Both work on any host; the steps
below compare `master` with the `phase4-gpu` branch on an NVIDIA GPU.

```bash
cargo install oha                                  # load generator used by run.sh
bench/run.sh http://127.0.0.1:8080                 # s1 s2 h1 r1
bench/run.sh http://127.0.0.1:8080 s1 s2           # a subset
DURATION=60s bench/run.sh http://127.0.0.1:8080    # longer runs, steadier numbers
```

| Scenario | Request | Clients | What it shows |
|---|---|---|---|
| `s1` | 1 short query | 64 | search traffic: per-request overhead, cross-request batching |
| `s2` | 32 passages of ~90 tokens | 8 | ingestion: raw compute |
| `h1` | 64 texts of 3–250 words | 4 | padding waste on mixed lengths |
| `r1` | rerank, 1 query + 1 document | 64 | reranker serving (needs a rerank model) |

Laptops throttle under sustained load: alternate the setups you compare
(A, B, A, B) and only trust differences larger than the spread between the
repeated runs.

## Testing `phase4-gpu` on a GPU

The branch targets accelerators; on CPU its changes are neutral (measured).
What it changes on a GPU, and how to switch each change off to isolate it:

| Commit | Change | Expected on GPU | Switch off |
|---|---|---|---|
| direct path sorted by length | rows of a big request run shortest-first, batches capped by padded tokens | less padding (`h1`; CPU: 5x with batching off) | compare with `master` |
| pooling in the graph | the runtime returns `[batch, dim]` instead of `[batch, tokens, dim]` | far less device-to-host copying (`s2`, long texts) | `pooling_in_graph: false` |
| fp16 on GPU | `dtype: auto` picks the published fp16 graph | ~2–3x compute on tensor cores | `dtype: fp32` |
| TensorRT | fp16 engines, explicit shape profile, engine/timing caches | one engine for all batch shapes, no rebuilds | compare `gpu-cuda.yaml` |
| batch pipelining | next batch's tensors built while the current one runs (2 workers per session) | less GPU idle time (`s1`) | compare with `master` |

### 1. Build both versions

On Linux with CUDA ≥ 13.2 and cuDNN 9 (TensorRT for the TensorRT profile):

```bash
git worktree add ../rs-infer-master master
(cd ../rs-infer-master && make gpu-trt)            # or make gpu-cuda
make gpu-trt                                       # this checkout, on phase4-gpu
```

### 2. Measure each version

```bash
./target/release/rsinfer-server --config bench/configs/gpu-cuda.yaml &
bench/run.sh http://127.0.0.1:8080 | tee phase4-cuda.txt
kill %1
(cd ../rs-infer-master && ./target/release/rsinfer-server --config ../rs-infer/bench/configs/gpu-cuda.yaml) &
bench/run.sh http://127.0.0.1:8080 | tee master-cuda.txt
kill %1
```

Repeat with `bench/configs/gpu-tensorrt.yaml` (the first requests wait for the
TensorRT engine build; `run.sh` warms up for 3 s, so run it twice and keep the
second). Watch GPU memory and utilization meanwhile with
`nvidia-smi dmon -s um`.

Check at startup that the GPU is really used: `GET /v1/models` lists the
execution providers, and the `model loaded` log line shows `precision` (the
fp16 graph is ~half the fp32 size in `model_mb`) and `pooling_in_graph=true`.
A GPU build on a host where CUDA libraries are missing logs
`ERROR ... register` and silently runs on CPU.

### 3. Check the results are the same

Run the CPU fp32 reference next to the GPU server, then compare:

```bash
./target/release/rsinfer-server --config bench/configs/cpu-reference.yaml &   # port 8081
python3 bench/compare.py http://127.0.0.1:8080 http://127.0.0.1:8081
```

Expect a minimum cosine ≥ 0.999 for fp16 on GPU (≥ 0.995 for int8 on CPU).

### What to report back

For each setup (`master` / `phase4-gpu`, CUDA / TensorRT): the `run.sh`
lines, the GPU model, peak GPU memory, and the `compare.py` line. To attribute
a gain to one commit, rerun `phase4-gpu` with that change switched off (table
above).
