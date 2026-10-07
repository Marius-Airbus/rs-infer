use std::{sync::Arc, time::Duration};

use crate::{
	config::Scoring,
	model::Meta,
	pipeline::{blocking, run_forward, softmax, Fwd},
	tokenize::make_inputs,
	Error, LoadedModel, Result,
};

#[derive(Debug, Clone, Copy)]
pub struct Scored {
	pub index: usize,
	pub score: f64,
}

pub struct RerankOutcome {
	pub scores: Vec<Scored>,
	pub prompt_tokens: usize,
}

/// Cross-encoder scoring of (query, document) pairs, in input order.
pub async fn score_pairs(model: &Arc<LoadedModel>, query: String, documents: Vec<String>, queue_wait: Duration) -> Result<RerankOutcome> {
	let pairs: Vec<(String, String)> = documents.into_iter().map(|d| (query.clone(), d)).collect();
	score_text_pairs(model, pairs, queue_wait).await
}

pub async fn score_text_pairs(model: &Arc<LoadedModel>, pairs: Vec<(String, String)>, queue_wait: Duration) -> Result<RerankOutcome> {
	let Meta::Rerank { scoring, yes_id, no_id, output } = &model.meta else {
		return Err(Error::KindMismatch {
			name: model.name().into(),
			expected: "rerank",
			actual: model.kind().as_str(),
		});
	};
	let (scoring, yes_id, no_id, output) = (*scoring, *yes_id, *no_id, output.clone());

	let m = Arc::clone(model);
	let enc = blocking(move || m.encoder.encode_pairs(&pairs)).await??;
	let token_count = enc.token_count();
	let attn = enc.attention_mask.clone();

	let pooled = model.pool.acquire(queue_wait).await?;
	let scores = pooled
		.run_blocking(move |session| -> Result<Vec<f64>> {
			let inputs = make_inputs(session, &enc)?;
			run_forward(session, inputs, &output, |fwd| apply_scoring(&fwd, scoring, yes_id, no_id, &attn))
		})
		.await?;

	Ok(RerankOutcome {
		scores: scores.iter().copied().enumerate().map(|(index, score)| Scored { index, score }).collect(),
		prompt_tokens: token_count,
	})
}

fn apply_scoring(fwd: &Fwd<'_>, mut scoring: Scoring, yes_id: Option<u32>, no_id: Option<u32>, attn: &[Vec<i64>]) -> Result<Vec<f64>> {
	match fwd.shape.as_slice() {
		[b, 1] => {
			// Default maps the single logit through sigmoid -> relevance_score in (0,1),
			// monotonic, so ranking is identical to raw logits. `logit` opts out.
			if scoring == Scoring::Auto {
				scoring = Scoring::Sigmoid;
			}
			Ok((0..*b)
				.map(|i| match scoring {
					Scoring::Sigmoid => 1.0 / (1.0 + (-(fwd.data[i] as f64)).exp()),
					_ => fwd.data[i] as f64,
				})
				.collect())
		}
		[b, 2] => {
			if matches!(scoring, Scoring::Auto | Scoring::Softmax) {
				scoring = Scoring::Softmax;
			}
			Ok((0..*b)
				.map(|i| {
					let row = &fwd.data[i * 2..(i + 1) * 2];
					match scoring {
						Scoring::Sigmoid => 1.0 / (1.0 + (-(row[1] as f64 - row[0] as f64)).exp()),
						_ => softmax(row)[1],
					}
				})
				.collect())
		}
		[b, s, v] => {
			// Vocabulary-distribution outputs (Qwen3-Reranker style: P("yes") at the last position).
			let (b, s, v) = (*b, *s, *v);
			let yes = yes_id.ok_or_else(|| Error::Config("yes_no scoring requires a tokenizer that maps 'yes'".into()))? as usize;
			let no = no_id.ok_or_else(|| Error::Config("yes_no scoring requires a tokenizer that maps 'no'".into()))? as usize;
			let mut out = Vec::with_capacity(b);
			for i in 0..b {
				let len = attn.get(i).map(|m| m.iter().sum::<i64>().max(1) as usize).unwrap_or(1).min(s);
				let pos = (i * s + len - 1) * v;
				let row = [fwd.data[pos + no], fwd.data[pos + yes]];
				out.push(softmax(&row)[1]);
			}
			Ok(out)
		}
		other => Err(Error::BadOutputShape(other.to_vec())),
	}
}

#[cfg(test)]
#[path = "../tests/pipeline/rerank_tests.rs"]
mod tests;
