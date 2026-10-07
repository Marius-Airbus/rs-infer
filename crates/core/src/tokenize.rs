use std::borrow::Cow;

use ort::{
	session::{Session, SessionInputs, SessionInputValue},
	value::{Outlet, Tensor, TensorElementType},
};
use tokenizers::{EncodeInput, Encoding, PaddingParams, PostProcessor, Tokenizer, TruncationParams, TruncationDirection};

use crate::{Error, Result};

/// One tokenized batch, padded to the longest row in the batch.
pub struct Encoded {
	pub input_ids: Vec<Vec<i64>>,
	pub attention_mask: Vec<Vec<i64>>,
	pub token_type_ids: Vec<Vec<i64>>,
	/// Character offsets per token ((0,0) for special/pad tokens); only populated
	/// by the `encode_*_offsets` variants (empty otherwise, to avoid the cost).
	pub offsets: Vec<Vec<(usize, usize)>>,
	pub batch: usize,
	pub seq: usize,
	/// Inputs longer than `max_len` that the tokenizer cut (it kept the overflow aside).
	pub truncated: usize,
}

impl Encoded {
	pub fn token_count(&self) -> usize {
		self.attention_mask.iter().map(|r| r.iter().map(|&m| m as usize).sum::<usize>()).sum()
	}

	/// Splits into batches of at most `max_rows` rows, each trimmed to its own
	/// longest row (padding is on the right).
	pub fn split(&self, max_rows: usize) -> Vec<Encoded> {
		let max_rows = max_rows.max(1);
		(0..self.batch)
			.step_by(max_rows)
			.map(|start| {
				let end = (start + max_rows).min(self.batch);
				let seq = self.attention_mask[start..end].iter().map(|m| m.iter().filter(|&&v| v != 0).count()).max().unwrap_or(0);
				fn cut<T: Clone>(rows: &[Vec<T>], seq: usize) -> Vec<Vec<T>> {
					rows.iter().map(|r| r[..seq.min(r.len())].to_vec()).collect()
				}
				Encoded {
					input_ids: cut(&self.input_ids[start..end], seq),
					attention_mask: cut(&self.attention_mask[start..end], seq),
					token_type_ids: cut(&self.token_type_ids[start..end], seq),
					offsets: self.offsets.get(start..end).map(|o| cut(o, seq)).unwrap_or_default(),
					batch: end - start,
					seq,
					truncated: 0,
				}
			})
			.collect()
	}

	/// Unpadded rows (right padding stripped via the attention mask), e.g. for
	/// the cross-request batcher.
	pub fn into_rows(self) -> Vec<Row> {
		self.input_ids
			.into_iter()
			.zip(self.token_type_ids)
			.zip(&self.attention_mask)
			.map(|((mut ids, mut type_ids), mask)| {
				let len = mask.iter().filter(|&&m| m != 0).count();
				ids.truncate(len);
				type_ids.truncate(len);
				Row { ids, type_ids }
			})
			.collect()
	}

	/// Right-padded batch of `rows`.
	pub fn from_rows(rows: &[Row]) -> Encoded {
		let seq = rows.iter().map(|r| r.ids.len()).max().unwrap_or(0);
		let pad = |v: &[i64]| {
			let mut v = v.to_vec();
			v.resize(seq, 0);
			v
		};
		Encoded {
			input_ids: rows.iter().map(|r| pad(&r.ids)).collect(),
			attention_mask: rows.iter().map(|r| pad(&vec![1; r.ids.len()])).collect(),
			token_type_ids: rows.iter().map(|r| pad(&r.type_ids)).collect(),
			offsets: Vec::new(),
			batch: rows.len(),
			seq,
			truncated: 0,
		}
	}
}

/// One unpadded model input row.
#[derive(Debug, Clone, PartialEq)]
pub struct Row {
	pub ids: Vec<i64>,
	/// Segment ids (0 for the first text of a pair, 1 for the second).
	pub type_ids: Vec<i64>,
}

pub struct Encoder {
	tokenizer: Tokenizer,
	max_len: Option<usize>,
	pub vocab_size: usize,
}

impl Encoder {
	/// `stride`: tokens shared by consecutive windows when a text overflows
	/// `max_len` (used by [`Encoder::encode_texts_windows`]; 0 elsewhere).
	pub fn new(path: &std::path::Path, max_len: Option<usize>, stride: usize) -> Result<Self> {
		// tokenizers' encode_batch fans out on rayon by default; that pool then
		// oversubscribes against ORT's intra-op threads. Serialize it unless the
		// operator explicitly configured TOKENIZERS_PARALLELISM.
		static PAR_INIT: std::sync::Once = std::sync::Once::new();
		PAR_INIT.call_once(|| {
			if !tokenizers::parallelism::is_parallelism_configured() {
				tokenizers::parallelism::set_parallelism(false);
			}
		});
		let mut tokenizer = Tokenizer::from_file(path).map_err(|e| Error::Tokenize(format!("{}: {e}", path.display())))?;
		tokenizer.with_padding(Some(PaddingParams::default()));
		if let Some(max) = max_len {
			tokenizer
				.with_truncation(Some(TruncationParams {
					max_length: max,
					stride,
					direction: TruncationDirection::Right,
					..Default::default()
				}))
				.map_err(|e| Error::Tokenize(e.to_string()))?;
		}
		let vocab = tokenizer.get_vocab(true).len();
		Ok(Self { tokenizer, max_len, vocab_size: vocab })
	}

	/// Tokens left for a document next to `query` in one pair input (`max_len`
	/// minus the query and the pair's special tokens), at least a quarter of
	/// `max_len`. `None` when no `max_len` is configured.
	pub fn doc_chunk_budget(&self, query: &str) -> Result<Option<usize>> {
		let Some(max) = self.max_len else { return Ok(None) };
		let query_tokens = self.tokenizer.encode(query, false).map_err(|e| Error::Tokenize(e.to_string()))?.len();
		let specials = self.tokenizer.get_post_processor().map_or(0, |p| p.added_tokens(true));
		Ok(Some(max.saturating_sub(query_tokens + specials).max(max / 4).max(1)))
	}

	/// Splits `text` into consecutive chunks of at most `max_tokens` tokens, cut at
	/// token boundaries. Always returns at least one chunk.
	pub fn split_text(&self, text: &str, max_tokens: usize) -> Result<Vec<String>> {
		let mut enc = self.tokenizer.encode(text, false).map_err(|e| Error::Tokenize(e.to_string()))?;
		// Byte offsets of every content token; inputs over max_len come back as
		// overflow windows, possibly overlapping (stride), so skip repeats.
		let mut offsets: Vec<(usize, usize)> = Vec::new();
		let overflow = enc.take_overflowing();
		for part in std::iter::once(&enc).chain(&overflow) {
			for &(s, e) in part.get_offsets() {
				if e > s && s >= offsets.last().map_or(0, |o| o.1) {
					offsets.push((s, e));
				}
			}
		}
		if offsets.is_empty() {
			return Ok(vec![text.to_string()]);
		}
		Ok(offsets
			.chunks(max_tokens.max(1))
			.map(|c| text.get(c[0].0..c[c.len() - 1].1).unwrap_or(text).to_string())
			.collect())
	}

	pub fn single_token_id(&self, word: &str) -> Option<u32> {
		let enc = self.tokenizer.encode(word, false).ok()?;
		enc.get_ids().first().copied()
	}

	fn from_encodings(encodings: Vec<Encoding>, with_offsets: bool) -> Encoded {
		// Rows are normally batch-padded already; overflow windows may not be, so
		// right-pad everything to the longest row.
		let seq = encodings.iter().map(|x| x.len()).max().unwrap_or(0);
		let pad = |mut v: Vec<i64>| {
			v.resize(seq, 0);
			v
		};
		let mut e = Encoded {
			input_ids: Vec::with_capacity(encodings.len()),
			attention_mask: Vec::with_capacity(encodings.len()),
			token_type_ids: Vec::with_capacity(encodings.len()),
			offsets: Vec::new(),
			batch: encodings.len(),
			seq,
			truncated: encodings.iter().filter(|x| !x.get_overflowing().is_empty()).count(),
		};
		for enc in &encodings {
			e.input_ids.push(pad(enc.get_ids().iter().map(|&i| i as i64).collect()));
			e.attention_mask.push(pad(enc.get_attention_mask().iter().map(|&m| m as i64).collect()));
			e.token_type_ids.push(pad(enc.get_type_ids().iter().map(|&t| t as i64).collect()));
			if with_offsets {
				let mut offsets = enc.get_offsets().to_vec();
				offsets.resize(seq, (0, 0));
				e.offsets.push(offsets);
			}
		}
		e
	}

	pub fn encode_texts(&self, texts: &[String]) -> Result<Encoded> {
		let inputs: Vec<EncodeInput<'_>> = texts.iter().map(|t| EncodeInput::from(Cow::Borrowed(t.as_str()))).collect();
		let encodings = self
			.tokenizer
			.encode_batch(inputs, true)
			.map_err(|e| Error::Tokenize(e.to_string()))?;
		Ok(Self::from_encodings(encodings, false))
	}

	pub fn encode_texts_offsets(&self, texts: &[String]) -> Result<Encoded> {
		let inputs: Vec<EncodeInput<'_>> = texts.iter().map(|t| EncodeInput::from(Cow::Borrowed(t.as_str()))).collect();
		let encodings = self
			.tokenizer
			.encode_batch_char_offsets(inputs, true)
			.map_err(|e| Error::Tokenize(e.to_string()))?;
		Ok(Self::from_encodings(encodings, true))
	}

	/// Like [`Encoder::encode_texts_offsets`], but a text longer than `max_len`
	/// yields one row per overlapping window (the tokenizer's overflow, `stride`
	/// tokens shared) instead of being cut. Also returns each row's text index.
	pub fn encode_texts_windows(&self, texts: &[String]) -> Result<(Encoded, Vec<usize>)> {
		let inputs: Vec<EncodeInput<'_>> = texts.iter().map(|t| EncodeInput::from(Cow::Borrowed(t.as_str()))).collect();
		let encodings = self
			.tokenizer
			.encode_batch_char_offsets(inputs, true)
			.map_err(|e| Error::Tokenize(e.to_string()))?;
		let mut rows = Vec::with_capacity(encodings.len());
		let mut owners = Vec::with_capacity(encodings.len());
		for (i, mut enc) in encodings.into_iter().enumerate() {
			let overflow = enc.take_overflowing();
			rows.push(enc);
			owners.push(i);
			for window in overflow {
				rows.push(window);
				owners.push(i);
			}
		}
		Ok((Self::from_encodings(rows, true), owners))
	}

	pub fn encode_pairs(&self, pairs: &[(String, String)]) -> Result<Encoded> {
		let inputs: Vec<EncodeInput<'_>> = pairs
			.iter()
			.map(|(a, b)| EncodeInput::Dual(Cow::Borrowed(a.as_str()).into(), Cow::Borrowed(b.as_str()).into()))
			.collect();
		let encodings = self
			.tokenizer
			.encode_batch(inputs, true)
			.map_err(|e| Error::Tokenize(e.to_string()))?;
		Ok(Self::from_encodings(encodings, false))
	}

	pub fn encode_pairs_offsets(&self, pairs: &[(String, String)]) -> Result<Encoded> {
		let inputs: Vec<EncodeInput<'_>> = pairs
			.iter()
			.map(|(a, b)| EncodeInput::Dual(Cow::Borrowed(a.as_str()).into(), Cow::Borrowed(b.as_str()).into()))
			.collect();
		let encodings = self
			.tokenizer
			.encode_batch_char_offsets(inputs, true)
			.map_err(|e| Error::Tokenize(e.to_string()))?;
		Ok(Self::from_encodings(encodings, true))
	}
}

/// Builds the token id/mask/type tensors for a session, intersecting the tokenizer
/// outputs with the names the model actually declares as inputs.
pub fn make_inputs(session: &Session, enc: &Encoded) -> Result<SessionInputs<'static, 'static>> {
	let mut flat_ids = Vec::with_capacity(enc.batch * enc.seq);
	for row in &enc.input_ids {
		flat_ids.extend_from_slice(row);
	}
	let shape = vec![enc.batch as i64, enc.seq as i64];

	let mut map: Vec<(Cow<'static, str>, ort::session::SessionInputValue<'static>)> = Vec::with_capacity(3);
	let mut pushed = Vec::with_capacity(3);
	for input in session.inputs() {
		let name = input.name();
		pushed.push(name.to_string());
		// Decoder-only exports (e.g. Qwen3-Embedding): a single-pass embedding
		// forward runs with an empty KV cache.
		if name.starts_with("past_key_values.") {
			map.push((Cow::Owned(name.to_string()), empty_kv_cache(input, enc.batch)?));
			continue;
		}
		let tensor = match name {
			"input_ids" => Tensor::from_array((shape.clone(), std::mem::take(&mut flat_ids)))?,
			"attention_mask" => Tensor::from_array((shape.clone(), enc.attention_mask.iter().flatten().copied().collect::<Vec<i64>>()))?,
			"token_type_ids" => Tensor::from_array((shape.clone(), enc.token_type_ids.iter().flatten().copied().collect::<Vec<i64>>()))?,
			// HF convention: position_ids = cumsum(attention_mask) - 1.
			"position_ids" => Tensor::from_array((shape.clone(), position_ids(&enc.attention_mask)))?,
			other => {
				return Err(Error::Ort(ort::Error::new(format!(
					"model requires unsupported input '{other}'; required inputs: {pushed:?}"
				))));
			}
		};
		map.push((Cow::Owned(name.to_string()), tensor.into()));
	}
	Ok(SessionInputs::ValueMap(map))
}

/// HF convention: position_ids = cumsum(attention_mask) - 1 along the sequence dim.
fn position_ids(mask: &[Vec<i64>]) -> Vec<i64> {
	let mut out = Vec::new();
	for row in mask {
		let mut acc = 0i64;
		for &m in row {
			acc += m;
			out.push(acc - 1);
		}
	}
	out
}

/// Empty KV-cache tensor for a `past_key_values.N.{key,value}` input of a
/// decoder-only export: shape [batch, num_kv_heads, 0, head_dim], derived from
/// the declared input shape (dynamic dims are -1 in ORT).
///
/// The export's attention-mask graph only broadcasts correctly with an empty
/// cache (past=0), so a single-pass embedding forward must pass a zero-length
/// cache. Note: CoreML EP rejects zero-element tensors; such models require
/// the CPU EP.
fn empty_kv_cache(input: &Outlet, batch: usize) -> Result<SessionInputValue<'static>> {
	let declared = input
		.dtype()
		.tensor_shape()
		.ok_or_else(|| Error::Ort(ort::Error::new(format!("input '{}' is not a tensor", input.name()))))?;
	let dims: Vec<i64> = declared.iter().copied().collect();
	if dims.len() != 4 {
		return Err(Error::Ort(ort::Error::new(format!(
			"unexpected past_key_values shape for '{}': {dims:?}",
			input.name()
		))));
	}
	let shape = vec![batch as i64, dims[1], 0, dims[3]];
	match input.dtype().tensor_type() {
		Some(TensorElementType::Float32) => Ok(Tensor::from_array((shape, Vec::<f32>::new()))?.into()),
		Some(other) => Err(Error::Ort(ort::Error::new(format!(
			"unsupported KV-cache dtype {other:?} for '{}'",
			input.name()
		)))),
		None => Err(Error::Ort(ort::Error::new(format!("input '{}' has no element type", input.name())))),
	}
}

#[cfg(test)]
#[path = "tests/tokenize_tests.rs"]
mod tests;
