//! Unit tests for [`tokenize`](super).

use super::{Encoded, Encoder};
use crate::test_fixtures::word_tokenizer;

/// Right-padded batch: one row per entry of `lens` (real tokens), padded to `seq`.
fn encoded(lens: &[usize], seq: usize) -> Encoded {
	let row = |n: usize, v: i64| {
		let mut r = vec![v; n];
		r.resize(seq, 0);
		r
	};
	Encoded {
		input_ids: lens.iter().map(|&n| row(n, 7)).collect(),
		attention_mask: lens.iter().map(|&n| row(n, 1)).collect(),
		token_type_ids: lens.iter().map(|&n| row(n, 0)).collect(),
		offsets: Vec::new(),
		batch: lens.len(),
		seq,
		truncated: 0,
	}
}

fn strings(v: &[&str]) -> Vec<String> {
	v.iter().map(|s| s.to_string()).collect()
}

#[test]
fn split_text_cuts_long_text_into_token_chunks() {
	// max_len 6 < 10 words: the tokenizer's overflow windows are joined back first.
	let enc = Encoder::new(&word_tokenizer(), Some(6), 0).unwrap();
	let text = (0..10).map(|i| format!("w{i}")).collect::<Vec<_>>().join(" ");
	assert_eq!(enc.split_text(&text, 4).unwrap(), ["w0 w1 w2 w3", "w4 w5 w6 w7", "w8 w9"]);
	assert_eq!(enc.split_text("", 4).unwrap(), [""]);
}

#[test]
fn doc_chunk_budget_leaves_room_for_query_and_specials() {
	let enc = Encoder::new(&word_tokenizer(), Some(16), 0).unwrap();
	// 16 - 2 query tokens - 3 pair specials ([CLS] q [SEP] d [SEP]).
	assert_eq!(enc.doc_chunk_budget("q1 q2").unwrap(), Some(11));
	// A query that eats the window still leaves a quarter of it for the document.
	assert_eq!(enc.doc_chunk_budget(&"q ".repeat(20)).unwrap(), Some(4));
	assert_eq!(Encoder::new(&word_tokenizer(), None, 0).unwrap().doc_chunk_budget("q").unwrap(), None);
}

#[test]
fn counts_truncated_inputs() {
	let enc = Encoder::new(&word_tokenizer(), Some(6), 0).unwrap();
	// 3 words + [CLS]/[SEP] fit in 6 tokens; 8 words do not.
	let texts = strings(&["a b c", "a b c d e f g h"]);
	assert_eq!(enc.encode_texts(&texts).unwrap().truncated, 1);
	let (rows, truncated) = enc.encode_rows(&texts).unwrap();
	assert_eq!((rows[0].len(), rows[1].len(), truncated), (5, 6, 1));
	let pairs = vec![("q".to_string(), "a b".to_string()), ("q".to_string(), "a b c d e f g".to_string())];
	assert_eq!(enc.encode_pairs(&pairs).unwrap().truncated, 1);
	assert_eq!(enc.encode_texts(&strings(&["a", "b c"])).unwrap().truncated, 0);
}

#[test]
fn split_bounds_rows_and_trims_padding() {
	let enc = encoded(&[2, 5, 3, 1, 4], 5);
	let parts = enc.split(2);
	assert_eq!(parts.iter().map(|p| p.batch).collect::<Vec<_>>(), [2, 2, 1]);
	assert_eq!(parts.iter().map(|p| p.seq).collect::<Vec<_>>(), [5, 3, 4]);
	assert_eq!(parts[1].attention_mask, vec![vec![1, 1, 1], vec![1, 0, 0]]);
	assert_eq!(parts.iter().map(|p| p.token_count()).sum::<usize>(), enc.token_count());
}

#[test]
fn split_keeps_a_small_batch_whole() {
	let enc = encoded(&[2, 3], 3);
	let parts = enc.split(32);
	assert_eq!(parts.len(), 1);
	assert_eq!(parts[0].input_ids, enc.input_ids);
}
