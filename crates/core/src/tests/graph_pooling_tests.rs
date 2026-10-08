//! Unit tests for [`graph_pooling`](super).

use std::{
	path::{Path, PathBuf},
	sync::atomic::{AtomicUsize, Ordering},
};

use ort::{session::Session, value::Tensor};

use super::{cached_pooled, output_names, POOLED_OUTPUT};
use crate::{
	config::Pooling,
	pipeline::{embedding::pool_rows, Fwd},
	quantize::pb,
};

const VOCAB: usize = 50;
const DIM: usize = 8;

fn temp_dir(tag: &str) -> PathBuf {
	static N: AtomicUsize = AtomicUsize::new(0);
	let dir = std::env::temp_dir().join(format!("rsinfer-pool-{tag}-{}-{}", std::process::id(), N.fetch_add(1, Ordering::SeqCst)));
	let _ = std::fs::remove_dir_all(&dir);
	std::fs::create_dir_all(&dir).unwrap();
	dir
}

fn value_info(name: &str, elem_type: u64) -> Vec<u8> {
	let mut tensor_type = Vec::new();
	pb::put_uint(&mut tensor_type, 1, elem_type);
	let mut type_proto = Vec::new();
	pb::put_len(&mut type_proto, 1, &tensor_type);
	let mut vi = Vec::new();
	pb::put_len(&mut vi, 1, name.as_bytes());
	pb::put_len(&mut vi, 2, &type_proto);
	vi
}

fn embedding_table() -> Vec<f32> {
	(0..VOCAB * DIM).map(|i| ((i * 37 % 101) as f32 - 50.0) / 25.0).collect()
}

/// `last_hidden_state = Gather(E[VOCAB,DIM], input_ids)`, with an `attention_mask`
/// input; `E` embedded, or in an external `model.onnx_data` file.
fn token_model(dir: &Path, opset: u64, external: bool) -> PathBuf {
	let data: Vec<u8> = embedding_table().iter().flat_map(|v| v.to_le_bytes()).collect();
	let mut e = Vec::new();
	pb::put_uint(&mut e, 1, VOCAB as u64);
	pb::put_uint(&mut e, 1, DIM as u64);
	pb::put_uint(&mut e, 2, 1);
	pb::put_len(&mut e, 8, b"E");
	if external {
		std::fs::write(dir.join("model.onnx_data"), &data).unwrap();
		for (k, v) in [("location", "model.onnx_data".to_string()), ("offset", "0".into()), ("length", data.len().to_string())] {
			let mut entry = Vec::new();
			pb::put_len(&mut entry, 1, k.as_bytes());
			pb::put_len(&mut entry, 2, v.as_bytes());
			pb::put_len(&mut e, 13, &entry);
		}
		pb::put_uint(&mut e, 14, 1);
	} else {
		pb::put_len(&mut e, 9, &data);
	}
	let mut node = Vec::new();
	pb::put_len(&mut node, 1, b"E");
	pb::put_len(&mut node, 1, b"input_ids");
	pb::put_len(&mut node, 2, b"last_hidden_state");
	pb::put_len(&mut node, 4, b"Gather");
	let mut g = Vec::new();
	pb::put_len(&mut g, 1, &node);
	pb::put_len(&mut g, 2, b"tokens");
	pb::put_len(&mut g, 5, &e);
	pb::put_len(&mut g, 11, &value_info("input_ids", 7));
	pb::put_len(&mut g, 11, &value_info("attention_mask", 7));
	pb::put_len(&mut g, 12, &value_info("last_hidden_state", 1));
	let mut opset_import = Vec::new();
	pb::put_len(&mut opset_import, 1, b"");
	pb::put_uint(&mut opset_import, 2, opset);
	let mut m = Vec::new();
	pb::put_uint(&mut m, 1, 8);
	pb::put_len(&mut m, 8, &opset_import);
	pb::put_len(&mut m, 7, &g);
	let path = dir.join("model.onnx");
	std::fs::write(&path, m).unwrap();
	path
}

/// Right-padded batch: rows of 3 and 5 real tokens.
const IDS: [[i64; 5]; 2] = [[4, 9, 2, 0, 0], [7, 1, 3, 8, 6]];
const MASK: [[i64; 5]; 2] = [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]];

fn run(model: &Path, output: &str) -> (Vec<usize>, Vec<f32>) {
	crate::ep::init_shared_thread_pool(2).unwrap();
	let mut session = Session::builder().unwrap().commit_from_file(model).unwrap();
	let ids = Tensor::from_array((vec![2i64, 5], IDS.concat())).unwrap();
	let mask = Tensor::from_array((vec![2i64, 5], MASK.concat())).unwrap();
	let outputs = session.run(ort::inputs!["input_ids" => ids, "attention_mask" => mask]).unwrap();
	let (shape, data) = outputs[output].try_extract_tensor::<f32>().unwrap();
	(shape.iter().map(|&d| d as usize).collect(), data.to_vec())
}

fn check(opset: u64, external: bool, poolings: &[Pooling]) {
	let src_dir = temp_dir("src");
	let src = token_model(&src_dir, opset, external);
	let (shape, tokens) = run(&src, "last_hidden_state");
	let mask: Vec<Vec<i64>> = MASK.iter().map(|r| r.to_vec()).collect();
	for &pooling in poolings {
		let expected = pool_rows(&Fwd { shape: shape.clone(), data: &tokens }, pooling, &mask).unwrap();
		let pooled = cached_pooled(&src, "last_hidden_state", pooling, &temp_dir("cache")).unwrap();
		assert!(output_names(&pooled).unwrap().contains(&POOLED_OUTPUT.to_string()));
		let (out_shape, out) = run(&pooled, POOLED_OUTPUT);
		assert_eq!(out_shape, [2, DIM], "{pooling:?}");
		for (row, want) in out.chunks(DIM).zip(&expected) {
			for (a, b) in row.iter().zip(want) {
				assert!((a - b).abs() < 1e-5, "{pooling:?} opset {opset}: {row:?} vs {want:?}");
			}
		}
	}
}

const ALL: [Pooling; 3] = [Pooling::Mean, Pooling::Cls, Pooling::Last];

#[test]
fn pooled_output_matches_cpu_pooling() {
	check(17, false, &ALL);
}

#[test]
fn pooled_output_matches_with_axes_attributes() {
	// Before opset 13, Unsqueeze/ReduceSum take `axes` as an attribute.
	check(12, false, &ALL);
	// Opset 11 exports (e.g. Xenova's e5): mean and CLS only.
	check(11, false, &[Pooling::Mean, Pooling::Cls]);
}

#[test]
fn external_weights_are_linked_not_rewritten() {
	check(17, true, &ALL);
}

#[test]
fn graphs_without_attention_mask_or_output_are_refused() {
	let dir = temp_dir("refuse");
	let src = token_model(&dir, 17, false);
	assert!(cached_pooled(&src, "nope", Pooling::Mean, &temp_dir("refuse-cache")).is_err());
	let opset11 = token_model(&temp_dir("old"), 11, false);
	assert!(cached_pooled(&opset11, "last_hidden_state", Pooling::Last, &temp_dir("old-cache")).is_err());
	let opset10 = token_model(&temp_dir("older"), 10, false);
	assert!(cached_pooled(&opset10, "last_hidden_state", Pooling::Mean, &temp_dir("older-cache")).is_err());
}
