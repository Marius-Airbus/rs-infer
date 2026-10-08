use std::{collections::HashMap, sync::Arc, time::Duration};

use serde::Serialize;

use ort::session::{Session, SessionInputs};

use crate::{
	model::{Meta, OutSel},
	pipeline::{blocking, run_forward, run_rows, Extract, Fwd, RowOut},
	tokenize::Encoded,
	Error, LoadedModel, Result,
};

#[derive(Debug, Clone, Serialize)]
pub struct Entity {
	/// The detected PII type, e.g. `person` (from `B-person`).
	pub entity_type: String,
	pub text: String,
	/// Mean token probability over the span.
	pub score: f64,
	/// Character offsets into the input text.
	pub start: usize,
	pub end: usize,
}

#[derive(Debug)]
pub struct DetectOutput {
	pub entities: Vec<Vec<Entity>>,
	pub tokens: usize,
}

/// Runs token classification and decodes BIO/BIOES spans into entities.
pub async fn detect(model: &Arc<LoadedModel>, texts: Vec<String>, threshold: Option<f64>, queue_wait: Duration) -> Result<DetectOutput> {
	let Meta::Pii { id2label, .. } = &model.meta else {
		return Err(Error::KindMismatch {
			name: model.name().into(),
			expected: "pii",
			actual: model.kind().as_str(),
		});
	};
	let id2label = id2label.clone();
	let threshold = threshold.unwrap_or(model.cfg.threshold).clamp(0.0, 1.0);

	// Texts longer than max_len come back as several overlapping windows (rows),
	// so nothing past the first max_len tokens escapes detection.
	let m = Arc::clone(model);
	let (enc, owners, texts) = blocking(move || m.encoder.encode_texts_windows(&texts).map(|(enc, owners)| (enc, owners, texts))).await??;
	let token_count = enc.token_count();
	let offsets = enc.offsets.clone();
	let per_token = run_rows(model, enc, queue_wait).await?.into_iter().map(RowOut::into_tokens).collect::<Result<Vec<_>>>()?;
	let mut windows: Vec<Vec<Window>> = vec![Vec::new(); texts.len()];
	for ((row, row_offsets), owner) in per_token.into_iter().zip(offsets).zip(owners) {
		// Content tokens only: special and pad tokens have empty (0,0) offsets.
		windows[owner].push(row.into_iter().zip(row_offsets).filter(|(_, (s, e))| e > s).unzip());
	}
	let results = texts
		.iter()
		.zip(windows)
		.map(|(text, text_windows)| {
			let (preds, token_offsets) = stitch_windows(text_windows);
			decode_entities(text, &preds, &token_offsets, &id2label, threshold)
		})
		.collect();
	Ok(DetectOutput { entities: results, tokens: token_count })
}

/// One window's content tokens: (label id, probability) and char offsets.
type Window = (Vec<(usize, f64)>, Vec<(usize, usize)>);

/// Joins one text's overlapping windows (in order) into a single token sequence.
/// A token in an overlap is taken from the window where it sits farther from the
/// edge, i.e. with context on both sides: the first half of the overlap from the
/// earlier window, the second half from the later one.
fn stitch_windows(windows: Vec<Window>) -> Window {
	let (mut preds, mut offsets): Window = (Vec::new(), Vec::new());
	for (w_preds, w_offsets) in windows {
		let covered = offsets.last().map_or(0, |o| o.1);
		let overlap = w_offsets.iter().take_while(|o| o.0 < covered).count().min(preds.len());
		let from_earlier = overlap / 2;
		let keep = preds.len() - (overlap - from_earlier);
		preds.truncate(keep);
		offsets.truncate(keep);
		preds.extend_from_slice(&w_preds[from_earlier..]);
		offsets.extend_from_slice(&w_offsets[from_earlier..]);
	}
	(preds, offsets)
}

/// Batch extractor for token-classification models: per token, the argmax label
/// and its probability.
pub(crate) fn extractor(n_labels: usize, output: OutSel) -> Arc<Extract> {
	Arc::new(move |session: &mut Session, inputs: SessionInputs<'static, 'static>, _: &Encoded| -> Result<Vec<RowOut>> {
		let rows = run_forward(session, inputs, &output, |fwd| argmax_probs(&fwd, n_labels))?;
		Ok(rows.into_iter().map(RowOut::Tokens).collect())
	})
}

/// Per token: (predicted label id, softmax probability of that label).
fn argmax_probs(fwd: &Fwd<'_>, num_labels: usize) -> Result<Vec<Vec<(usize, f64)>>> {
	match fwd.shape.as_slice() {
		[b, s, c] => {
			let n = *c;
			if n != num_labels {
				tracing::warn!(model_labels = num_labels, output_labels = n, "output label count differs from config id2label");
			}
			let (b, s) = (*b, *s);
			let mut out = Vec::with_capacity(b);
			for i in 0..b {
				let mut row = Vec::with_capacity(s);
				for step in 0..s {
					let base = (i * s + step) * n;
					let logits = &fwd.data[base..base + n];
					let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
					let mut best = (0usize, f32::NEG_INFINITY);
					let mut sum = 0f32;
					for (j, l) in logits.iter().enumerate() {
						let e = *l - max;
						sum += e.exp();
						if e > best.1 {
							best = (j, e);
						}
					}
					row.push((best.0, (best.1.exp() / sum) as f64));
				}
				out.push(row);
			}
			Ok(out)
		}
		other => Err(Error::BadOutputShape(other.to_vec())),
	}
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Tag {
	Other,
	Begin(usize),
	Inside(usize),
	End(usize),
	Single(usize),
}

/// label string -> Tag; `types` interns entity-type names.
fn parse_tag(label: &str, types: &mut Vec<String>, index: &mut HashMap<String, usize>) -> Tag {
	if label == "O" || label.is_empty() {
		return Tag::Other;
	}
	let (prefix, ty) = match label.split_once('-') {
		Some((p, t)) => (p, t),
		None => ("", label),
	};
	let idx = match index.get(ty) {
		Some(i) => *i,
		None => {
			let i = types.len();
			types.push(ty.to_string());
			index.insert(ty.to_string(), i);
			i
		}
	};
	match prefix {
		"B" => Tag::Begin(idx),
		"I" => Tag::Inside(idx),
		"E" => Tag::End(idx),
		"S" => Tag::Single(idx),
		_ => Tag::Inside(idx),
	}
}

#[derive(Debug, Clone, Copy)]
struct OpenSpan {
	ty: usize,
	start: usize,
	end: usize,
	score_sum: f64,
	tokens: usize,
}

/// Decode one row's tags into char-span entities (HF `simple`-style aggregation, BIOES aware).
fn decode_entities(text: &str, row: &[(usize, f64)], token_offsets: &[(usize, usize)], id2label: &[String], threshold: f64) -> Vec<Entity> {
	let chars: Vec<char> = text.chars().collect();
	let mut types: Vec<String> = Vec::new();
	let mut index: HashMap<String, usize> = HashMap::new();
	let tags: Vec<Tag> = row
		.iter()
		.map(|(id, prob)| {
			if *prob < threshold {
				return Tag::Other;
			}
			let label = id2label.get(*id).map(String::as_str).unwrap_or("O");
			parse_tag(label, &mut types, &mut index)
		})
		.collect();

	let mut entities = Vec::new();
	let mut open: Option<OpenSpan> = None;
	for (i, tag) in tags.iter().enumerate() {
		let (s, e) = token_offsets.get(i).copied().unwrap_or((0, 0));
		let content_token = e > s;
		let score = row[i].1;
		let finish = |open: &mut Option<OpenSpan>, entities: &mut Vec<Entity>| {
			if let Some(sp) = open.take() {
				push_span(entities, &chars, &types, sp);
			}
		};
		match *tag {
			Tag::Other => {
				finish(&mut open, &mut entities);
			}
			Tag::Single(ty) if content_token => {
				finish(&mut open, &mut entities);
				push_span(&mut entities, &chars, &types, OpenSpan { ty, start: s, end: e, score_sum: score, tokens: 1 });
			}
			Tag::Begin(ty) if content_token => {
				finish(&mut open, &mut entities);
				open = Some(OpenSpan { ty, start: s, end: e, score_sum: score, tokens: 1 });
			}
			Tag::Inside(ty) if content_token => match &mut open {
				Some(sp) if sp.ty == ty => {
					sp.end = sp.end.max(e);
					sp.score_sum += score;
					sp.tokens += 1;
				}
				_ => {
					finish(&mut open, &mut entities);
					open = Some(OpenSpan { ty, start: s, end: e, score_sum: score, tokens: 1 });
				}
			},
			Tag::End(ty) if content_token => match &mut open {
				Some(sp) if sp.ty == ty => {
					sp.end = sp.end.max(e);
					sp.score_sum += score;
					sp.tokens += 1;
					finish(&mut open, &mut entities);
				}
				_ => {
					finish(&mut open, &mut entities);
					push_span(&mut entities, &chars, &types, OpenSpan { ty, start: s, end: e, score_sum: score, tokens: 1 });
				}
			},
			_ => {}
		}
	}
	if let Some(sp) = open {
		push_span(&mut entities, &chars, &types, sp);
	}
	entities
}

fn push_span(out: &mut Vec<Entity>, chars: &[char], types: &[String], sp: OpenSpan) {
	let (mut s, mut e) = (sp.start, sp.end);
	while s < e && chars.get(s).is_some_and(|c| c.is_whitespace()) {
		s += 1;
	}
	while e > s && chars.get(e - 1).is_some_and(|c| c.is_whitespace()) {
		e -= 1;
	}
	if e <= s {
		return;
	}
	out.push(Entity {
		entity_type: types.get(sp.ty).cloned().unwrap_or_default(),
		text: chars[s..e].iter().collect(),
		score: sp.score_sum / sp.tokens.max(1) as f64,
		start: s,
		end: e,
	});
}

/// Mask detected entities. `Remove` deletes spans, `Mask` replaces each char with `mask_char`.
pub fn redact(text: &str, entities: &[Entity], mode: RedactMode, mask_char: char) -> String {
	let chars: Vec<char> = text.chars().collect();
	let mut sorted: Vec<&Entity> = entities.iter().collect();
	sorted.sort_by_key(|e| e.start);
	let mut out = String::with_capacity(text.len());
	let mut pos = 0usize;
	for ent in sorted {
		if ent.start < pos {
			continue; // overlapping span already redacted
		}
		out.extend(chars[pos..ent.start.min(chars.len())].iter());
		match mode {
			RedactMode::Mask => {
				for _ in ent.start..ent.end {
					out.push(mask_char);
				}
			}
			RedactMode::Remove => {}
		}
		pos = ent.end.max(pos).min(chars.len());
	}
	out.extend(chars[pos..].iter());
	out
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RedactMode {
	Mask,
	Remove,
}

#[cfg(test)]
#[path = "../tests/pipeline/pii_tests.rs"]
mod tests;
