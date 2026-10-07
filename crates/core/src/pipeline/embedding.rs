use std::{sync::Arc, time::Duration};

use crate::{
	config::Pooling,
	model::Meta,
	pipeline::{blocking, forward_rows, run_forward, warn_truncated, Fwd},
	Error, LoadedModel, Result,
};

#[derive(Debug)]
pub struct EmbedOutput {
	pub vectors: Vec<Vec<f32>>,
	pub tokens: usize,
}

/// Produces one embedding vector per input text, in request order.
pub async fn embed(model: &Arc<LoadedModel>, texts: Vec<String>, queue_wait: Duration) -> Result<EmbedOutput> {
	let Meta::Embedding { pooling, output, normalize, dimensions } = &model.meta else {
		return Err(Error::KindMismatch {
			name: model.name().into(),
			expected: "embedding",
			actual: model.kind().as_str(),
		});
	};
	let (pooling, output, normalize, dimensions) = (*pooling, output.clone(), *normalize, *dimensions);

	if let Some(batcher) = &model.batcher {
		let m = Arc::clone(model);
		let (rows, truncated) = blocking(move || m.encoder.encode_rows(&texts)).await??;
		warn_truncated(model, truncated);
		let token_count = rows.iter().map(|r| r.len()).sum();
		let vectors = batcher.submit(rows, queue_wait).await?;
		tracing::trace!(model = model.name(), rows = vectors.len(), tokens = token_count, "embedding done (batched)");
		return Ok(EmbedOutput { vectors, tokens: token_count });
	}

	let m = Arc::clone(model);
	let enc = blocking(move || m.encoder.encode_texts(&texts)).await??;
	warn_truncated(model, enc.truncated);
	let token_count = enc.token_count();
	let max_batch = model.cfg.max_batch;

	let pooled = model.pool.acquire(queue_wait).await?;
	let vectors = pooled
		.run_blocking(move |session| -> Result<Vec<Vec<f32>>> {
			forward_rows(session, &enc, max_batch, &output, |fwd, part| pool_rows(&fwd, pooling, &part.attention_mask))
		})
		.await?;

	let n = vectors.len();
	let mut out = Vec::with_capacity(n);
	for mut vec in vectors {
		if let Some(d) = dimensions {
			if d < vec.len() {
				vec.truncate(d);
			}
		}
		if normalize {
			l2_normalize(&mut vec);
		}
		out.push(vec);
	}
	tracing::trace!(model = model.name(), rows = n, tokens = token_count, "embedding done");
	Ok(EmbedOutput { vectors: out, tokens: token_count })
}

/// Embeddings from raw token id rows (OpenAI `input` as token arrays).
pub async fn embed_tokens(model: &Arc<LoadedModel>, rows: Vec<Vec<u32>>, queue_wait: Duration) -> Result<EmbedOutput> {
	let Meta::Embedding { pooling, output, normalize, dimensions } = &model.meta else {
		return Err(Error::KindMismatch {
			name: model.name().into(),
			expected: "embedding",
			actual: model.kind().as_str(),
		});
	};
	let (pooling, output, normalize, dimensions) = (*pooling, output.clone(), *normalize, *dimensions);
	if rows.is_empty() {
		return Ok(EmbedOutput { vectors: Vec::new(), tokens: 0 });
	}
	let tokens: usize = rows.iter().map(|r| r.len()).sum();

	if let Some(batcher) = &model.batcher {
		let vectors = batcher.submit(rows, queue_wait).await?;
		return Ok(EmbedOutput { vectors, tokens });
	}

	let max_batch = model.cfg.max_batch.max(1);
	let pooled = model.pool.acquire(queue_wait).await?;
	let vectors = pooled
		.run_blocking(move |session| -> Result<Vec<Vec<f32>>> {
			let mut out = Vec::with_capacity(rows.len());
			for chunk in rows.chunks(max_batch) {
				let inputs = crate::tokenize::make_token_inputs(session, chunk)?;
				let attn: Vec<Vec<i64>> = chunk.iter().map(|r| vec![1i64; r.len()]).collect();
				out.extend(run_forward(session, inputs, &output, |fwd| pool_rows(&fwd, pooling, &attn))?);
			}
			Ok(out)
		})
		.await?;

	let mut out = Vec::with_capacity(vectors.len());
	for mut vec in vectors {
		if let Some(d) = dimensions {
			if d < vec.len() {
				vec.truncate(d);
			}
		}
		if normalize {
			l2_normalize(&mut vec);
		}
		out.push(vec);
	}
	Ok(EmbedOutput { vectors: out, tokens })
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
