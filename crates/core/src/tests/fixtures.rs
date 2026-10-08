//! Shared test fixtures.

use std::{
	path::PathBuf,
	sync::atomic::{AtomicUsize, Ordering},
};

/// Writes a tiny BERT-style `tokenizer.json` (whitespace split, word-level vocab,
/// `[CLS] a [SEP] b [SEP]` post-processing) and returns its path. Every word is
/// `[UNK]`: tests care about token counts and offsets, not ids.
pub(crate) fn word_tokenizer() -> PathBuf {
	static N: AtomicUsize = AtomicUsize::new(0);
	let json = r#"{
		"version": "1.0",
		"truncation": null,
		"padding": null,
		"added_tokens": [],
		"normalizer": null,
		"pre_tokenizer": { "type": "WhitespaceSplit" },
		"post_processor": { "type": "BertProcessing", "sep": ["[SEP]", 3], "cls": ["[CLS]", 2] },
		"decoder": null,
		"model": { "type": "WordLevel", "vocab": { "[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3 }, "unk_token": "[UNK]" }
	}"#;
	let dir = std::env::temp_dir().join(format!("rsinfer-tests-{}", std::process::id()));
	std::fs::create_dir_all(&dir).unwrap();
	let path = dir.join(format!("tokenizer-{}.json", N.fetch_add(1, Ordering::SeqCst)));
	std::fs::write(&path, json).unwrap();
	path
}

/// Minimal ONNX model builders for tests.
pub(crate) mod onnx {
	use crate::quantize::pb;

	/// A graph input/output declaration with an element type and no shape.
	pub(crate) fn value_info(name: &str, elem_type: u64) -> Vec<u8> {
		let mut tensor_type = Vec::new();
		pb::put_uint(&mut tensor_type, 1, elem_type);
		let mut type_proto = Vec::new();
		pb::put_len(&mut type_proto, 1, &tensor_type);
		let mut vi = Vec::new();
		pb::put_len(&mut vi, 1, name.as_bytes());
		pb::put_len(&mut vi, 2, &type_proto);
		vi
	}

	pub(crate) fn node(op: &str, inputs: &[&str], outputs: &[&str], int_attrs: &[(&str, i64)]) -> Vec<u8> {
		let mut n = Vec::new();
		for i in inputs {
			pb::put_len(&mut n, 1, i.as_bytes());
		}
		for o in outputs {
			pb::put_len(&mut n, 2, o.as_bytes());
		}
		pb::put_len(&mut n, 4, op.as_bytes());
		for (name, v) in int_attrs {
			let mut a = Vec::new();
			pb::put_len(&mut a, 1, name.as_bytes());
			pb::put_uint(&mut a, 3, *v as u64);
			pb::put_uint(&mut a, 20, 2);
			pb::put_len(&mut n, 5, &a);
		}
		n
	}

	/// ModelProto bytes (IR 8, default-domain opset `opset`).
	pub(crate) fn model(opset: u64, nodes: &[Vec<u8>], inputs: &[Vec<u8>], outputs: &[Vec<u8>]) -> Vec<u8> {
		let mut g = Vec::new();
		for n in nodes {
			pb::put_len(&mut g, 1, n);
		}
		pb::put_len(&mut g, 2, b"test");
		for i in inputs {
			pb::put_len(&mut g, 11, i);
		}
		for o in outputs {
			pb::put_len(&mut g, 12, o);
		}
		let mut opset_import = Vec::new();
		pb::put_len(&mut opset_import, 1, b"");
		pb::put_uint(&mut opset_import, 2, opset);
		let mut m = Vec::new();
		pb::put_uint(&mut m, 1, 8);
		pb::put_len(&mut m, 8, &opset_import);
		pb::put_len(&mut m, 7, &g);
		m
	}
}
