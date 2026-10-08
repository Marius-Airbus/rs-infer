//! Unit tests for [`tokenize`](super).

use super::{Encoded, Encoder, Row};
use crate::test_fixtures::word_tokenizer;

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
	let pairs = vec![("q".to_string(), "a b".to_string()), ("q".to_string(), "a b c d e f g".to_string())];
	assert_eq!(enc.encode_pairs(&pairs).unwrap().truncated, 1);
	assert_eq!(enc.encode_texts(&strings(&["a", "b c"])).unwrap().truncated, 0);
}

#[test]
fn rows_strip_padding_and_rebatch() {
	let enc = Encoder::new(&word_tokenizer(), Some(16), 0).unwrap();
	let pairs = vec![("q".to_string(), "a b c".to_string()), ("q".to_string(), "a".to_string())];
	let batch = enc.encode_pairs(&pairs).unwrap();
	let rows = batch.into_rows();
	// [CLS] q [SEP] a b c [SEP] / [CLS] q [SEP] a [SEP]: padding gone, segments kept.
	assert_eq!(rows.iter().map(|r| r.ids.len()).collect::<Vec<_>>(), [7, 5]);
	assert_eq!(rows[1].type_ids, [0, 0, 0, 1, 1]);
	let rebatched = Encoded::from_rows(&rows);
	assert_eq!((rebatched.batch, rebatched.seq, rebatched.token_count()), (2, 7, 12));
	assert_eq!(rebatched.attention_mask[1], [1, 1, 1, 1, 1, 0, 0]);
	assert_eq!(rebatched.into_rows(), rows);
}

#[test]
fn from_rows_pads_token_type_ids() {
	let rows = [Row { ids: vec![5, 6], type_ids: vec![0, 1] }, Row { ids: vec![7], type_ids: vec![0] }];
	let enc = Encoded::from_rows(&rows);
	assert_eq!(enc.token_type_ids, vec![vec![0, 1], vec![0, 0]]);
	assert_eq!(enc.input_ids, vec![vec![5, 6], vec![7, 0]]);
}
