//! Unit tests for [`tokenize`](super).

use super::Encoded;

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
	}
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
