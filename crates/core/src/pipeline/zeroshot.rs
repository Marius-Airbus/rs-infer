use std::{sync::Arc, time::Duration};

use ort::session::{Session, SessionInputs};

use crate::{
	model::{Meta, OutSel},
	pipeline::{blocking, run_forward, run_rows, softmax, warn_truncated, Extract, RowOut},
	tokenize::Encoded,
	Error, LoadedModel, Result,
};

#[derive(Debug, Clone)]
pub struct ClassOutcome {
	pub labels: Vec<String>,
	pub scores: Vec<f64>,
}

#[derive(Debug)]
pub struct ClassOutput {
	pub outcomes: Vec<ClassOutcome>,
	pub tokens: usize,
}

/// NLI-based zero-shot classification over encoder-only MNLI models.
///
/// For each text, one (text, hypothesis) pair per candidate label is scored; the
/// entailment/contradiction logits are compared per hypothesis and aggregated:
/// single-label → softmax over labels, multi-label → sigmoid per label (HF-compatible).
pub async fn classify(model: &Arc<LoadedModel>, texts: Vec<String>, candidates: Vec<String>, multi_label: bool, queue_wait: Duration) -> Result<ClassOutput> {
	let Meta::Zeroshot { entailment, contradiction, template, .. } = &model.meta else {
		return Err(Error::KindMismatch {
			name: model.name().into(),
			expected: "zeroshot",
			actual: model.kind().as_str(),
		});
	};
	let (entailment, contradiction, template) = (*entailment, *contradiction, template.clone());
	if candidates.is_empty() {
		return Err(Error::BadRequest("candidate_labels must not be empty".into()));
	}
	let n_labels = candidates.len();

	let mut pairs = Vec::with_capacity(texts.len() * n_labels);
	for text in &texts {
		for label in &candidates {
			let hypothesis = template.replace("{text}", text).replace("{label}", label).replace("{}", label);
			pairs.push((text.clone(), hypothesis));
		}
	}

	let m = Arc::clone(model);
	let enc = blocking(move || m.encoder.encode_pairs(&pairs)).await??;
	warn_truncated(model, enc.truncated);
	let token_count = enc.token_count();
	let per_class = entailment_logits(&logit_rows(model, enc, queue_wait).await?, entailment, contradiction)?;

	let mut outcomes = Vec::with_capacity(texts.len());
	for text_idx in 0..texts.len() {
		let slice = &per_class[text_idx * n_labels..(text_idx + 1) * n_labels];
		let scores: Vec<f64> = if multi_label {
			slice.iter().map(|l| 1.0 / (1.0 + (-l).exp())).collect()
		} else {
			softmax_f64(slice)
		};
		let mut ranked: Vec<(String, f64)> = candidates.iter().cloned().zip(scores).collect();
		ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
		outcomes.push(ClassOutcome {
			labels: ranked.iter().map(|(l, _)| l.clone()).collect(),
			scores: ranked.iter().map(|(_, s)| *s).collect(),
		});
	}
	Ok(ClassOutput { outcomes, tokens: token_count })
}

async fn logit_rows(model: &Arc<LoadedModel>, enc: Encoded, queue_wait: Duration) -> Result<Vec<Vec<f32>>> {
	run_rows(model, enc, queue_wait).await?.into_iter().map(RowOut::into_logits).collect()
}

/// Batch extractor for NLI models: the raw class logits of each (premise, hypothesis) row.
pub(crate) fn extractor(output: OutSel) -> Arc<Extract> {
	Arc::new(move |session: &mut Session, inputs: SessionInputs<'static, 'static>, _: &Encoded| -> Result<Vec<RowOut>> {
		run_forward(session, inputs, &output, |fwd| match fwd.shape.as_slice() {
			[_, k] => Ok(fwd.data.chunks(*k).map(|row| RowOut::Logits(row.to_vec())).collect()),
			other => Err(Error::BadOutputShape(other.to_vec())),
		})
	})
}

/// entailment-minus-contradiction logit per pair row.
fn entailment_logits(rows: &[Vec<f32>], entailment: usize, contradiction: usize) -> Result<Vec<f64>> {
	rows.iter().map(|row| Ok((label(row, entailment)? - label(row, contradiction)?) as f64)).collect()
}

fn label(row: &[f32], id: usize) -> Result<f32> {
	row.get(id).copied().ok_or_else(|| Error::Config(format!("label id {id} outside output dim {}", row.len())))
}

/// Auxiliary verbs that can be fronted in a yes/no question.
const AUXES: &[&str] = &["is", "are", "was", "were", "do", "does", "did", "can", "could", "will", "would", "should", "may", "might", "must", "has", "have", "had"];

/// Rephrase a yes/no question into its affirmative assertion:
/// "Is this email important?" -> "This email is important."
///
/// Heuristic subject span: two words (one if the rest has exactly two tokens).
/// Garbled splits actually score *high* under MNLI entailment, so there is
/// deliberately no best-of-N fallback; non-invertible questions require an
/// explicit `assertion` from the caller.
pub fn invert_question(q: &str) -> Option<String> {
	let q = q.trim().trim_end_matches(['?', '.']);
	let words: Vec<&str> = q.split_whitespace().collect();
	if words.len() < 3 {
		return None;
	}
	let aux = words[0].to_ascii_lowercase();
	if !AUXES.iter().any(|a| *a == aux) {
		return None;
	}
	let rest = &words[1..];
	let subj_len = if rest.len() == 2 { 1 } else { 2 };
	if subj_len >= rest.len() {
		return None;
	}
	let (subj, pred) = rest.split_at(subj_len);
	let mut s = subj.join(" ");
	// Byte length of the first char: slicing at 1 panics on non-ASCII (e.g. "élan").
	let first_len = s.chars().next().map_or(0, char::len_utf8);
	let first = s[..first_len].to_uppercase();
	s.replace_range(..first_len, &first);
	Some(format!("{s} {aux} {}.", pred.join(" ")))
}

#[derive(Debug)]
pub struct TrueFalseOutput {
	/// P(entailment) of the chosen assertion per input, in (0,1).
	pub probabilities: Vec<f64>,
	/// The assertion actually judged (echoed for transparency).
	pub assertions: Vec<String>,
	pub tokens: usize,
}

/// NLI boolean classification: does the input entail the affirmative assertion
/// derived from `question` (or the caller-provided `assertion`)?
/// P(true) is the single entailment probability -- NOT a normalized
/// entail-vs-negation ratio: MNLI models rarely entail explicit negations, so
/// a rival "not" hypothesis would saturate the score at 1.0.
pub async fn classify_true_false(model: &Arc<LoadedModel>, inputs: Vec<String>, question: Option<String>, assertion: Option<String>, queue_wait: Duration) -> Result<TrueFalseOutput> {
	let Meta::Zeroshot { entailment, contradiction, .. } = &model.meta else {
		return Err(Error::KindMismatch {
			name: model.name().into(),
			expected: "zeroshot",
			actual: model.kind().as_str(),
		});
	};
	let (entailment, contradiction) = (*entailment, *contradiction);

	// Assertion per input: explicit, or the rephrased question.
	let assertion = match assertion {
		Some(a) if !a.trim().is_empty() => a,
		_ => {
			let q = question.as_deref().unwrap_or("");
			invert_question(q).ok_or_else(|| {
				Error::BadRequest(format!(
					"cannot phrase '{q}' as an assertion (not a simple yes/no question); pass an explicit `assertion` (e.g. \"This email is important.\")"
				))
			})?
		}
	};
	let mut pairs = Vec::with_capacity(inputs.len());
	for input in &inputs {
		pairs.push((input.clone(), assertion.clone()));
	}

	let m = Arc::clone(model);
	let enc = blocking(move || m.encoder.encode_pairs(&pairs)).await??;
	warn_truncated(model, enc.truncated);
	let token_count = enc.token_count();
	let probabilities = entailment_probs(&logit_rows(model, enc, queue_wait).await?, entailment, contradiction)?;
	let assertions = vec![assertion; inputs.len()];
	Ok(TrueFalseOutput { probabilities, assertions, tokens: token_count })
}

/// P(entailment) per row, contrasting only entailment vs contradiction (the
/// neutral class is ignored: MNLI models put most mass there for unrelated
/// pairs, which would swamp the entailment signal in a full-row softmax).
fn entailment_probs(rows: &[Vec<f32>], entailment: usize, contradiction: usize) -> Result<Vec<f64>> {
	rows.iter().map(|row| Ok(softmax(&[label(row, contradiction)?, label(row, entailment)?])[1])).collect()
}

pub(crate) fn softmax_f64(v: &[f64]) -> Vec<f64> {
	let max = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
	let exps: Vec<f64> = v.iter().map(|x| (x - max).exp()).collect();
	let sum: f64 = exps.iter().sum();
	exps.iter().map(|e| e / sum).collect()
}

#[cfg(test)]
#[path = "../tests/pipeline/zeroshot_tests.rs"]
mod tests;
