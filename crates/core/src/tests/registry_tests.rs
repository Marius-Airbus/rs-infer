//! Unit tests for [`registry`](super).

use super::*;

fn eps(v: &[&str]) -> Vec<String> {
	v.iter().map(|s| s.to_string()).collect()
}

#[test]
fn auto_dtype_prefers_fp16_graphs_on_gpu() {
	let auto = ModelConfig::default_for_test(None);
	assert_eq!(graph_preference(&auto, &eps(&["cuda", "cpu"])).dtype, Dtype::Fp16);
	assert_eq!(graph_preference(&auto, &eps(&["tensorrt", "cuda", "cpu"])).dtype, Dtype::Fp16);
	// CPU (int8 rewrite of the fp32 graph) and other accelerators keep `auto`.
	assert_eq!(graph_preference(&auto, &eps(&["cpu"])).dtype, Dtype::Auto);
	assert_eq!(graph_preference(&auto, &eps(&["openvino", "cpu"])).dtype, Dtype::Auto);
	// An explicit dtype is never overridden.
	let fp32 = ModelConfig { dtype: Dtype::Fp32, ..ModelConfig::default_for_test(None) };
	assert_eq!(graph_preference(&fp32, &eps(&["cuda", "cpu"])).dtype, Dtype::Fp32);
}
