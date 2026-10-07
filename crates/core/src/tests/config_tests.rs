//! Unit tests for [`config`](super).

use super::*;

#[test]
fn parses_example_config() {
	let yaml_src = r#"
server:
  bind: 0.0.0.0:8080
  max_queue: 128

models:
  - name: mini-lm
    kind: embedding
    hf: Xenova/all-MiniLM-L6-v2
    max_len: 256
    replicas: 2
    eps: [coreml, cpu]
    coreml_compute_units: cpu_and_gpu
    pooling: mean
    normalize: true
  - name: rerank
    kind: rerank
    path: models/rerank
    scoring: softmax
  - name: pii
    kind: pii
    hf: onnx-community/something
    threshold: 0.6
"#;
	let cfg: Config = serde_norway::from_str(yaml_src).unwrap();
	assert_eq!(cfg.server.max_queue, 128);
	assert_eq!(cfg.models.len(), 3);
	assert_eq!(cfg.models[0].eps, vec![EpName::Coreml, EpName::Cpu]);
	assert_eq!(cfg.models[0].coreml_compute_units, CoreMlComputeUnits::CpuAndGpu);
	assert_eq!(cfg.models[1].kind, Kind::Rerank);
	assert_eq!(cfg.models[2].threshold, 0.6);
	assert_eq!(cfg.models[0].batching, None);
	assert_eq!(cfg.models[0].hypothesis_template, "The text is about {label}.");
	cfg.models.iter().for_each(|m| m.validate().unwrap());
}

#[test]
fn parses_batching_block_with_defaults() {
	let yaml_src = r#"
models:
  - name: x
    kind: embedding
    path: models/x
    batching: { max_rows: 32 }
"#;
	let cfg: Config = serde_norway::from_str(yaml_src).unwrap();
	let b = cfg.models[0].batching.unwrap();
	assert_eq!(b.max_rows, 32);
	assert_eq!(b.max_tokens, 4096);
	assert_eq!(b.queue_rows, 1024);
}

#[test]
fn rejects_unknown_batching_field() {
	let yaml_src = r#"
models:
  - name: x
    kind: embedding
    path: models/x
    batching: { nonsense: 1 }
"#;
	assert!(serde_norway::from_str::<Config>(yaml_src).is_err());
}

#[test]
fn max_batch_defaults_and_validates() {
	let yaml_src = r#"
models:
  - name: x
    kind: embedding
    path: models/x
  - name: y
    kind: rerank
    path: models/y
    max_batch: 0
"#;
	let cfg: Config = serde_norway::from_str(yaml_src).unwrap();
	assert_eq!(cfg.models[0].max_batch, 32);
	assert!(cfg.models[1].validate().is_err());
}

#[test]
fn rejects_unknown_field() {
	let yaml_src = r#"
models:
  - name: x
    kind: embedding
    path: models/x
    bogus: 1
"#;
	assert!(serde_norway::from_str::<Config>(yaml_src).is_err());
}
