//! Unit tests for [`quantize`](super).

use std::{
	path::{Path, PathBuf},
	sync::atomic::{AtomicUsize, Ordering},
};

use ort::{session::Session, value::Tensor};

use super::{cached_int8, compare, pb, quantize_columns, quantize_model, quantize_rows, QuantStats, MIN_GATHER_ELEMS};
use crate::pipeline::RowOut;

fn temp_dir(tag: &str) -> PathBuf {
	static N: AtomicUsize = AtomicUsize::new(0);
	let dir = std::env::temp_dir().join(format!("rsinfer-quant-{tag}-{}-{}", std::process::id(), N.fetch_add(1, Ordering::SeqCst)));
	let _ = std::fs::remove_dir_all(&dir);
	std::fs::create_dir_all(&dir).unwrap();
	dir
}

/// Deterministic pseudo-random values in [-1, 1).
fn values(n: usize, seed: u64) -> Vec<f32> {
	let mut s = seed;
	(0..n)
		.map(|_| {
			s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
			((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
		})
		.collect()
}

fn tensor(name: &str, dims: &[i64], data: &[f32]) -> Vec<u8> {
	let mut t = Vec::new();
	for &d in dims {
		pb::put_uint(&mut t, 1, d as u64);
	}
	pb::put_uint(&mut t, 2, 1);
	pb::put_len(&mut t, 8, name.as_bytes());
	pb::put_len(&mut t, 9, &data.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
	t
}

fn node(op: &str, inputs: &[&str], outputs: &[&str]) -> Vec<u8> {
	let mut n = Vec::new();
	for i in inputs {
		pb::put_len(&mut n, 1, i.as_bytes());
	}
	for o in outputs {
		pb::put_len(&mut n, 2, o.as_bytes());
	}
	pb::put_len(&mut n, 4, op.as_bytes());
	n
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

/// `y = Gather(E[vocab,32], ids) . W[32,16]`, written to `dir/model.onnx`.
fn tiny_model(dir: &Path, opset: u64, vocab: usize) -> PathBuf {
	let mut g = Vec::new();
	pb::put_len(&mut g, 1, &node("Gather", &["E", "ids"], &["h"]));
	pb::put_len(&mut g, 1, &node("MatMul", &["h", "W"], &["y"]));
	pb::put_len(&mut g, 2, b"tiny");
	pb::put_len(&mut g, 5, &tensor("E", &[vocab as i64, 32], &values(vocab * 32, 1)));
	pb::put_len(&mut g, 5, &tensor("W", &[32, 16], &values(32 * 16, 2)));
	pb::put_len(&mut g, 11, &value_info("ids", 7));
	pb::put_len(&mut g, 12, &value_info("y", 1));
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

fn run(model: &Path, ids: &[i64]) -> Vec<f32> {
	crate::ep::init_shared_thread_pool(2).unwrap();
	let mut session = Session::builder().unwrap().commit_from_file(model).unwrap();
	let input = Tensor::from_array((vec![1i64, ids.len() as i64], ids.to_vec())).unwrap();
	let outputs = session.run(ort::inputs!["ids" => input]).unwrap();
	let (_, data) = outputs["y"].try_extract_tensor::<f32>().unwrap();
	data.to_vec()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
	let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
	dot / (a.iter().map(|x| x * x).sum::<f32>().sqrt() * b.iter().map(|x| x * x).sum::<f32>().sqrt())
}

#[test]
fn int8_rewrite_runs_in_onnx_runtime_and_tracks_fp32() {
	let vocab = MIN_GATHER_ELEMS / 32;
	let src_dir = temp_dir("src");
	let src = tiny_model(&src_dir, 17, vocab);
	let out = temp_dir("out");
	let stats = quantize_model(&src, &out, 127.0).unwrap();
	assert_eq!(stats, Some(QuantStats { matmuls: 1, gathers: 1 }));
	assert!(out.join("model.onnx_data").is_file(), "large int8 weights go to external data");

	let ids = [0, 7, 300, vocab as i64 - 1, 42];
	let (reference, quantized) = (run(&src, &ids), run(&out.join("model.onnx"), &ids));
	assert_eq!(reference.len(), quantized.len());
	for (row_ref, row_q) in reference.chunks(16).zip(quantized.chunks(16)) {
		assert!(cosine(row_ref, row_q) > 0.999, "row drifted: {}", cosine(row_ref, row_q));
	}
}

#[test]
fn nothing_to_quantize_is_reported() {
	let dir = temp_dir("old-opset");
	// DynamicQuantizeLinear needs opset 11.
	let src = tiny_model(&dir, 10, MIN_GATHER_ELEMS / 32);
	assert_eq!(quantize_model(&src, &temp_dir("old-opset-out"), 127.0).unwrap(), None);
}

#[test]
fn small_embedding_tables_stay_fp32() {
	let dir = temp_dir("small");
	let src = tiny_model(&dir, 17, 64);
	let stats = quantize_model(&src, &temp_dir("small-out"), 127.0).unwrap();
	assert_eq!(stats, Some(QuantStats { matmuls: 1, gathers: 0 }));
}

#[test]
fn cached_int8_builds_once() {
	let dir = temp_dir("cache-src");
	let src = tiny_model(&dir, 17, 64);
	let cache = temp_dir("cache");
	let first = cached_int8(&src, &cache).unwrap().expect("quantizable");
	let built = std::fs::metadata(&first).unwrap().modified().unwrap();
	let again = cached_int8(&src, &cache).unwrap().expect("cached");
	assert_eq!(first, again);
	assert_eq!(std::fs::metadata(&again).unwrap().modified().unwrap(), built);
}

#[test]
fn per_channel_scales_follow_each_column() {
	// W[2,3]: column maxima 2, 0.5, 0 -> scales 2/127, 0.5/127, 1 (all-zero column).
	let w = [2.0, -0.5, 0.0, -1.5, 0.2, 0.0];
	let (q, scales) = quantize_columns(&w, 2, 3, 127.0);
	assert_eq!(q, [127, -127, 0, -95, 51, 0]);
	assert!((scales[0] - 2.0 / 127.0).abs() < 1e-7 && (scales[1] - 0.5 / 127.0).abs() < 1e-7 && scales[2] == 1.0);
	// 7-bit range on CPUs without VNNI.
	assert_eq!(quantize_columns(&w, 2, 3, 63.0).0[..2], [63, -63]);
}

#[test]
fn per_row_scales_follow_each_embedding() {
	let (q, scales) = quantize_rows(&[1.2, -2.0, 0.5, 0.2], 2, 2);
	assert_eq!(q, [76, -127, 127, 51]);
	assert!((scales[0] - 2.0 / 127.0).abs() < 1e-7 && (scales[1] - 0.5 / 127.0).abs() < 1e-7);
}

#[test]
fn varints_round_trip() {
	for v in [0u64, 1, 127, 128, 300, u32::MAX as u64, (-1i64) as u64] {
		let mut buf = Vec::new();
		pb::put_varint(&mut buf, v);
		let mut pos = 0;
		assert_eq!(pb::varint(&buf, &mut pos).unwrap(), v);
		assert_eq!(pos, buf.len());
	}
	assert!(pb::fields(&[0x0a, 0x05, 0x61]).is_err(), "truncated length-delimited field");
}

#[test]
fn quality_check_thresholds_per_output_kind() {
	let v = |x: &[f32]| RowOut::Vector(x.to_vec());
	let ok = compare(&[v(&[1.0, 0.0])], &[v(&[0.999, 0.01])], &[]).unwrap();
	assert!(ok.pass && ok.metric == "cosine");
	assert!(!compare(&[v(&[1.0, 0.0])], &[v(&[0.9, 0.4])], &[]).unwrap().pass);
	assert!(!compare(&[RowOut::Score(0.9)], &[RowOut::Score(0.8)], &[]).unwrap().pass);
	// Token labels compared on real tokens only (mask), padding ignored.
	let tokens = |l: &[usize]| RowOut::Tokens(l.iter().map(|&x| (x, 0.9)).collect());
	let check = compare(&[tokens(&[0, 1, 2, 5])], &[tokens(&[0, 1, 2, 9])], &[vec![1, 1, 1, 0]]).unwrap();
	assert!(check.pass && check.worst == 1.0);
	assert!(compare(&[v(&[1.0])], &[RowOut::Score(1.0)], &[]).is_err());
}
