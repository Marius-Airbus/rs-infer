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
