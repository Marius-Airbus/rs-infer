//! Unit tests for [`pii`](super).

use super::*;
use crate::{test_fixtures::word_tokenizer, tokenize::Encoder};

#[test]
fn long_text_windows_stitch_back_to_every_token_once() {
	// max_len 8 = 6 words + [CLS]/[SEP] per window, 2 words of overlap.
	let enc = Encoder::new(&word_tokenizer(), Some(8), 2).unwrap();
	let text = (0..20).map(|i| format!("w{i}")).collect::<Vec<_>>().join(" ");
	let (encoded, owners) = enc.encode_texts_windows(std::slice::from_ref(&text)).unwrap();
	assert!(owners.len() > 1 && owners.iter().all(|&o| o == 0), "{owners:?}");
	let windows = encoded
		.offsets
		.iter()
		.map(|offs| {
			let offs: Vec<(usize, usize)> = offs.iter().copied().filter(|(s, e)| e > s).collect();
			(vec![(0, 1.0); offs.len()], offs)
		})
		.collect();
	let (_, offsets) = stitch_windows(windows);
	let words: Vec<&str> = offsets.iter().map(|&(s, e)| &text[s..e]).collect();
	assert_eq!(words, text.split(' ').collect::<Vec<_>>());
}

#[test]
fn entity_across_window_boundary_is_found_once() {
	let text = "a b c d e f g h";
	let offs = |r: std::ops::Range<usize>| r.map(|i| (2 * i, 2 * i + 1)).collect::<Vec<_>>();
	// Window 0 sees a..f, window 1 (2-token overlap) sees e..h; only window 1 has
	// the right-hand context to tag "f g" as a person.
	let w0 = (vec![(0, 0.9); 6], offs(0..6));
	let w1 = (vec![(0, 0.9), (1, 0.9), (2, 0.9), (0, 0.9)], offs(4..8));
	let (preds, offsets) = stitch_windows(vec![w0, w1]);
	assert_eq!(offsets, offs(0..8));
	let ents = decode_entities(text, &preds, &offsets, &labels(), 0.5);
	assert_eq!(ents.len(), 1, "{ents:?}");
	assert_eq!((ents[0].entity_type.as_str(), ents[0].text.as_str(), ents[0].start, ents[0].end), ("person", "f g", 10, 13));
}

fn labels() -> Vec<String> {
	vec![
		"O".into(),
		"B-person".into(),
		"I-person".into(),
		"B-email".into(),
		"I-email".into(),
		"B-location".into(),
		"I-location".into(),
	]
}

#[test]
fn decodes_bio_spans() {
	let text = "John Smith lives in Berlin";
	let offsets = vec![
		(0, 0),    // [CLS]
		(0, 4),    // John
		(5, 10),   // Smith
		(11, 16),  // lives
		(17, 19),  // in
		(20, 26),  // Berlin
		(0, 0),    // [SEP]
	];
	let row = vec![(0usize, 1.0), (1, 0.99), (2, 0.98), (0, 1.0), (0, 1.0), (5, 0.95), (0, 1.0)];
	let ents = decode_entities(text, &row, &offsets, &labels(), 0.5);
	assert_eq!(ents.len(), 2);
	assert_eq!(ents[0].entity_type, "person");
	assert_eq!(ents[0].text, "John Smith");
	assert_eq!(ents[0].start, 0);
	assert_eq!(ents[0].end, 10);
	assert_eq!(ents[1].entity_type, "location");
	assert_eq!(ents[1].text, "Berlin");
}

#[test]
fn decodes_bioes_single_and_end() {
	let text = "ada@x.io and Berlin";
	let labels: Vec<String> = ["O", "B-email", "I-email", "E-email", "S-phone", "B-location", "E-location"].into_iter().map(String::from).collect();
	let offsets = vec![(0, 0), (0, 3), (3, 4), (4, 8), (0, 0), (13, 19), (0, 0)];
	let row = vec![(0, 1.0), (1, 0.9), (2, 0.9), (3, 0.9), (0, 1.0), (5, 0.9), (0, 1.0)];
	let ents = decode_entities(text, &row, &offsets, &labels, 0.5);
	assert_eq!(ents.len(), 2);
	assert_eq!(ents[0].entity_type, "email");
	assert_eq!(ents[0].text, "ada@x.io");
	assert_eq!(ents[1].entity_type, "location");
	assert_eq!(ents[1].text, "Berlin");
}

#[test]
fn threshold_drops_low_confidence() {
	let text = "John";
	let offsets = vec![(0, 0), (0, 4), (0, 0)];
	let row = vec![(0, 1.0), (1, 0.3), (0, 1.0)];
	let ents = decode_entities(text, &row, &offsets, &labels(), 0.5);
	assert!(ents.is_empty());
}

#[test]
fn redacts_mask_and_remove() {
	let text = "call John Smith now";
	let e = Entity { entity_type: "person".into(), text: "John Smith".into(), score: 0.9, start: 5, end: 15 };
	assert_eq!(redact(text, std::slice::from_ref(&e), RedactMode::Mask, '*'), "call ********** now");
	assert_eq!(redact(text, std::slice::from_ref(&e), RedactMode::Remove, '*'), "call  now");
	let unicode = "héllo Wörld";
	let e2 = Entity { entity_type: "person".into(), text: "Wörld".into(), score: 0.9, start: 6, end: 11 };
	assert_eq!(redact(unicode, &[e2], RedactMode::Mask, '#'), "héllo #####");
}

#[test]
fn overlapping_redaction_applies_once() {
	let text = "abc def";
	let a = Entity { entity_type: "x".into(), text: "abc".into(), score: 0.9, start: 0, end: 3 };
	let b = Entity { entity_type: "y".into(), text: "abcd".into(), score: 0.9, start: 0, end: 4 };
	assert_eq!(redact(text, &[a, b], RedactMode::Mask, '*'), "*** def");
}
