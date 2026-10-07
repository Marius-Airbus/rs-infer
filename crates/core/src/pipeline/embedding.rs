use std::{sync::Arc, time::Duration};

use ort::session::Session;

use crate::{
	config::Pooling,
	model::{Meta, OutSel},
	pipeline::{blocking, forward, run_rows, warn_truncated, Extract, Fwd, RowOut},
	tokenize::{Encoded, Row},
	Error, LoadedModel, Result,
};

#[derive(Debug)]
pub struct EmbedOutput {
	pub vectors: Vec<Vec<f32>>,
	pub tokens: usize,
}

/// Produces one embedding vector per input text, in request order.
pub async fn embed(model: &Arc<LoadedModel>, texts: Vec<String>, queue_wait: Duration) -> Result<EmbedOutput> {
	ensure_embedding(model)?;
	let m = Arc::clone(model);
	let enc = blocking(move || m.encoder.encode_texts(&texts)).await??;
	warn_truncated(model, enc.truncated);
	let tokens = enc.token_count();
	let vectors = run_rows(model, enc, queue_wait).await?.into_iter().map(RowOut::into_vector).collect::<Result<_>>()?;
	Ok(EmbedOutput { vectors, tokens })
}

/// Embeddings from raw token id rows (OpenAI `input` as token arrays).
pub async fn embed_tokens(model: &Arc<LoadedModel>, rows: Vec<Vec<u32>>, queue_wait: Duration) -> Result<EmbedOutput> {
	ensure_embedding(model)?;
	if rows.is_empty() {
		return Ok(EmbedOutput { vectors: Vec::new(), tokens: 0 });
	}
	let rows: Vec<Row> = rows
		.into_iter()
		.map(|r| Row { type_ids: vec![0; r.len()], ids: r.into_iter().map(i64::from).collect() })
		.collect();
	let enc = Encoded::from_rows(&rows);
	let tokens = enc.token_count();
	let vectors = run_rows(model, enc, queue_wait).await?.into_iter().map(RowOut::into_vector).collect::<Result<_>>()?;
	Ok(EmbedOutput { vectors, tokens })
}

fn ensure_embedding(model: &LoadedModel) -> Result<()> {
	match model.meta {
		Meta::Embedding { .. } => Ok(()),
		_ => Err(Error::KindMismatch { name: model.name().into(), expected: "embedding", actual: model.kind().as_str() }),
	}
}

/// Batch extractor for embedding models: pooling, then Matryoshka truncation and
/// L2 normalization as configured.
pub(crate) fn extractor(pooling: Pooling, output: OutSel, normalize: bool, dimensions: Option<usize>) -> Arc<Extract> {
	Arc::new(move |session: &mut Session, enc: &Encoded| -> Result<Vec<RowOut>> {
		let rows = forward(session, enc, &output, |fwd| pool_rows(&fwd, pooling, &enc.attention_mask))?;
		Ok(rows
			.into_iter()
			.map(|mut v| {
				if let Some(d) = dimensions {
					v.truncate(d);
				}
				if normalize {
					l2_normalize(&mut v);
				}
				RowOut::Vector(v)
			})
			.collect())
	})
}

/// Convert a rank-3 tensor [B,T,D] (or rank-2 [B,D]) into [B,D] rows.
pub(crate) fn pool_rows(fwd: &Fwd<'_>, pooling: Pooling, attn: &[Vec<i64>]) -> Result<Vec<Vec<f32>>> {
	match fwd.shape.as_slice() {
		[b, d] => Ok((0..*b).map(|i| fwd.data[i * d..(i + 1) * d].to_vec()).collect()),
		[b, t, d] => {
			let (b, t, d) = (*b, *t, *d);
			let mut out = Vec::with_capacity(b);
			for i in 0..b {
				let base = &fwd.data[i * t * d..(i + 1) * t * d];
				let mask = attn.get(i).map(|m| m.as_slice()).unwrap_or(&[]);
				let row = match pooling {
					Pooling::Cls | Pooling::Auto => base[..d].to_vec(),
					Pooling::Mean => {
						let mut acc = vec![0f32; d];
						let mut n = 0f32;
						for (step, &token) in mask.iter().enumerate().take(t) {
							if token == 0 {
								continue;
							}
							n += 1.0;
							let off = step * d;
							for j in 0..d {
								acc[j] += base[off + j];
							}
						}
						if n == 0.0 {
							n = 1.0;
						}
						acc.iter_mut().for_each(|v| *v /= n);
						acc
					}
					Pooling::Last => {
						let mut last = 0usize;
						for (step, &token) in mask.iter().enumerate().take(t) {
							if token != 0 {
								last = step;
							}
						}
						base[last * d..(last + 1) * d].to_vec()
					}
				};
				out.push(row);
			}
			Ok(out)
		}
		other => Err(Error::BadOutputShape(other.to_vec())),
	}
}

pub(crate) fn l2_normalize(v: &mut [f32]) {
	let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
	if norm > 0.0 {
		v.iter_mut().for_each(|x| *x /= norm);
	}
}

#[cfg(test)]
#[path = "../tests/pipeline/embedding_tests.rs"]
mod tests;
