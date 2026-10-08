//! Unit tests for [`ep`](super).

use super::init_shared_thread_pool;

#[test]
fn shared_thread_pool_is_installed_once() {
	// Other tests building sessions install the pool too; whoever runs first wins.
	let n = init_shared_thread_pool(3).unwrap();
	assert!(n == 3 || n == 2, "unexpected pool size {n}");
	// Later calls keep the installed pool rather than failing or resizing it.
	assert_eq!(init_shared_thread_pool(5).unwrap(), n);
}

#[test]
fn tensorrt_profile_covers_every_batch_shape() {
	use crate::{config::ModelConfig, test_fixtures::onnx};

	let dir = std::env::temp_dir().join(format!("rsinfer-trt-{}", std::process::id()));
	std::fs::create_dir_all(&dir).unwrap();
	let write = |name: &str, inputs: &[&str]| {
		let path = dir.join(name);
		let ins: Vec<Vec<u8>> = inputs.iter().map(|i| onnx::value_info(i, 7)).collect();
		std::fs::write(&path, onnx::model(17, &[onnx::node("Identity", &["input_ids"], &["y"], &[])], &ins, &[onnx::value_info("y", 7)])).unwrap();
		path
	};
	let cfg = ModelConfig { max_len: Some(256), ..ModelConfig::default_for_test(None) }; // batching.max_rows 64 > max_batch 32
	let p = super::trt_profile(&write("enc.onnx", &["input_ids", "attention_mask"]), &cfg).unwrap();
	assert_eq!(p.min, "input_ids:1x1,attention_mask:1x1");
	assert_eq!(p.opt, "input_ids:32x128,attention_mask:32x128");
	assert_eq!(p.max, "input_ids:64x256,attention_mask:64x256");
	// Decoder exports (KV-cache inputs): left to TensorRT's dynamic handling.
	assert_eq!(super::trt_profile(&write("dec.onnx", &["input_ids", "attention_mask", "past_key_values.0.key"]), &cfg), None);
}
