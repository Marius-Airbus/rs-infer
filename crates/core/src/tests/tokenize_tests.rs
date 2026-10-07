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
fn counts_truncated_inputs() {
	let enc = Encoder::new(&word_tokenizer(), Some(6)).unwrap();
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
