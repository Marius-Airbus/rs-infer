use std::{sync::Arc, time::Duration};

use ort::session::{Session, SessionInputs};

use crate::{
	config::Scoring,
	model::{Meta, OutSel},
	pipeline::{blocking, run_forward, run_rows, softmax, warn_truncated, Extract, Fwd, RowOut},
	tokenize::Encoded,
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
///
/// With `max_chunks_per_doc`, a document too long for the model window is split
/// into chunks that each fit next to the query (at most that many, from the
/// start) and scores as its best chunk (Cohere semantics). Otherwise long
/// documents are truncated.
pub async fn score_pairs(model: &Arc<LoadedModel>, query: String, documents: Vec<String>, max_chunks_per_doc: Option<usize>, queue_wait: Duration) -> Result<RerankOutcome> {
	let Some(max_chunks) = max_chunks_per_doc.filter(|&n| n > 0) else {
		let pairs: Vec<(String, String)> = documents.into_iter().map(|d| (query.clone(), d)).collect();
		return score_text_pairs(model, pairs, queue_wait).await;
	};
	let n_docs = documents.len();
	let m = Arc::clone(model);
	let (pairs, owners) = blocking(move || -> Result<ChunkPairs> {
		let budget = m.encoder.doc_chunk_budget(&query)?;
		let (mut pairs, mut owners) = (Vec::new(), Vec::new());
		for (i, doc) in documents.into_iter().enumerate() {
			let chunks = match budget {
				Some(b) => m.encoder.split_text(&doc, b)?,
				None => vec![doc],
			};
			for chunk in chunks.into_iter().take(max_chunks) {
				pairs.push((query.clone(), chunk));
				owners.push(i);
			}
		}
		Ok((pairs, owners))
	})
	.await??;
	let outcome = score_text_pairs(model, pairs, queue_wait).await?;
	Ok(RerankOutcome { scores: best_per_doc(&outcome.scores, &owners, n_docs), prompt_tokens: outcome.prompt_tokens })
}

/// (query, chunk) pairs to score and, per pair, the index of its document.
type ChunkPairs = (Vec<(String, String)>, Vec<usize>);

/// Best chunk score per document, in document order.
fn best_per_doc(chunks: &[Scored], owners: &[usize], n_docs: usize) -> Vec<Scored> {
	let mut best: Vec<Scored> = (0..n_docs).map(|index| Scored { index, score: f64::NEG_INFINITY }).collect();
	for chunk in chunks {
		let doc = &mut best[owners[chunk.index]];
		doc.score = doc.score.max(chunk.score);
	}
	best
}

pub async fn score_text_pairs(model: &Arc<LoadedModel>, pairs: Vec<(String, String)>, queue_wait: Duration) -> Result<RerankOutcome> {
	if !matches!(model.meta, Meta::Rerank { .. }) {
		return Err(Error::KindMismatch {
			name: model.name().into(),
			expected: "rerank",
			actual: model.kind().as_str(),
		});
	}
	let m = Arc::clone(model);
	let enc = blocking(move || m.encoder.encode_pairs(&pairs)).await??;
	warn_truncated(model, enc.truncated);
	let prompt_tokens = enc.token_count();
	let scores = run_rows(model, enc, queue_wait).await?.into_iter().map(RowOut::into_score).collect::<Result<Vec<_>>>()?;
	Ok(RerankOutcome {
		scores: scores.into_iter().enumerate().map(|(index, score)| Scored { index, score }).collect(),
		prompt_tokens,
	})
}

/// Batch extractor for rerank models: one relevance score per (query, document) row.
pub(crate) fn extractor(scoring: Scoring, yes_id: Option<u32>, no_id: Option<u32>, output: OutSel) -> Arc<Extract> {
	Arc::new(move |session: &mut Session, inputs: SessionInputs<'static, 'static>, enc: &Encoded| -> Result<Vec<RowOut>> {
		let scores = run_forward(session, inputs, &output, |fwd| apply_scoring(&fwd, scoring, yes_id, no_id, &enc.attention_mask))?;
		Ok(scores.into_iter().map(RowOut::Score).collect())
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
