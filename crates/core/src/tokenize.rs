use std::borrow::Cow;

use ort::{
	session::{Session, SessionInputs, SessionInputValue},
	value::{Outlet, Tensor, TensorElementType},
};
use tokenizers::{EncodeInput, Encoding, PaddingParams, Tokenizer, TruncationParams, TruncationDirection};

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
}

pub struct Encoder {
	tokenizer: Tokenizer,
	pub vocab_size: usize,
}

impl Encoder {
	pub fn new(path: &std::path::Path, max_len: Option<usize>) -> Result<Self> {
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
					stride: 0,
					direction: TruncationDirection::Right,
					..Default::default()
				}))
				.map_err(|e| Error::Tokenize(e.to_string()))?;
		}
		let vocab = tokenizer.get_vocab(true).len();
		Ok(Self { tokenizer, vocab_size: vocab })
	}

	pub fn single_token_id(&self, word: &str) -> Option<u32> {
		let enc = self.tokenizer.encode(word, false).ok()?;
		enc.get_ids().first().copied()
	}

	fn from_encodings(encodings: Vec<Encoding>, with_offsets: bool) -> Encoded {
		let mut e = Encoded {
			input_ids: Vec::with_capacity(encodings.len()),
			attention_mask: Vec::with_capacity(encodings.len()),
			token_type_ids: Vec::with_capacity(encodings.len()),
			offsets: Vec::new(),
			batch: encodings.len(),
			seq: encodings.first().map(|x| x.len()).unwrap_or(0),
			truncated: encodings.iter().filter(|x| !x.get_overflowing().is_empty()).count(),
		};
		for enc in &encodings {
			e.input_ids.push(enc.get_ids().iter().map(|&i| i as i64).collect());
			e.attention_mask.push(enc.get_attention_mask().iter().map(|&m| m as i64).collect());
			e.token_type_ids.push(enc.get_type_ids().iter().map(|&t| t as i64).collect());
			if with_offsets {
				e.offsets.push(enc.get_offsets().to_vec());
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

	/// Token ids per text, without batch padding (right-pad stripped via the
	/// attention mask), plus how many texts were truncated. For feeding into the
	/// cross-request batcher.
	pub fn encode_rows(&self, texts: &[String]) -> Result<(Vec<Vec<u32>>, usize)> {
		let inputs: Vec<EncodeInput<'_>> = texts.iter().map(|t| EncodeInput::from(Cow::Borrowed(t.as_str()))).collect();
		let encodings = self
			.tokenizer
			.encode_batch(inputs, true)
			.map_err(|e| Error::Tokenize(e.to_string()))?;
		let truncated = encodings.iter().filter(|enc| !enc.get_overflowing().is_empty()).count();
		let rows = encodings
			.iter()
			.map(|enc| {
				let mask = enc.get_attention_mask();
				let real = mask.iter().position(|&m| m == 0).unwrap_or(mask.len());
				enc.get_ids()[..real].to_vec()
			})
			.collect();
		Ok((rows, truncated))
	}

	pub fn encode_texts_offsets(&self, texts: &[String]) -> Result<Encoded> {
		let inputs: Vec<EncodeInput<'_>> = texts.iter().map(|t| EncodeInput::from(Cow::Borrowed(t.as_str()))).collect();
		let encodings = self
			.tokenizer
			.encode_batch_char_offsets(inputs, true)
			.map_err(|e| Error::Tokenize(e.to_string()))?;
		Ok(Self::from_encodings(encodings, true))
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

/// Builds raw token-id inputs for pre-tokenized inputs (OpenAI numeric token arrays).
pub fn make_token_inputs(session: &Session, token_rows: &[Vec<u32>]) -> Result<SessionInputs<'static, 'static>> {
	let seq = token_rows.iter().map(|r| r.len()).max().unwrap_or(0);
	let mask = token_rows.iter().map(|r| {
		let mut v = vec![1i64; r.len()];
		v.resize(seq, 0);
		v
	});
	let mut ids = Vec::new();
	for r in token_rows {
		let mut v: Vec<i64> = r.iter().map(|&i| i as i64).collect();
		v.resize(seq, 0);
		ids.extend_from_slice(&v);
	}
	let types = vec![0i64; ids.len()];
	let shape = vec![token_rows.len() as i64, seq as i64];

	let mut map: Vec<(Cow<'static, str>, ort::session::SessionInputValue<'static>)> = Vec::new();
	let mut pushed = Vec::new();
	let masks: Vec<Vec<i64>> = mask.collect();
	let flat_masks: Vec<i64> = masks.iter().flatten().copied().collect();
	for input in session.inputs() {
		let name = input.name();
		pushed.push(name.to_string());
		// Decoder-only exports (e.g. Qwen3-Embedding): a single-pass embedding
		// forward runs with an empty KV cache.
		if name.starts_with("past_key_values.") {
			map.push((Cow::Owned(name.to_string()), empty_kv_cache(input, token_rows.len())?));
			continue;
		}
		let data: Vec<i64> = match name {
			"input_ids" => ids.clone(),
			"attention_mask" => flat_masks.clone(),
			"token_type_ids" => types.clone(),
			// HF convention: position_ids = cumsum(attention_mask) - 1.
			"position_ids" => position_ids(&masks),
			other => {
				return Err(Error::Ort(ort::Error::new(format!(
					"model requires unsupported input '{other}'; required inputs: {pushed:?}"
				))));
			}
		};
		map.push((Cow::Owned(name.to_string()), Tensor::from_array((shape.clone(), data))?.into()));
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
