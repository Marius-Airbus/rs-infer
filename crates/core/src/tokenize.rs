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

/// The inputs a model's graph takes, read once from a session so batches can be
/// turned into input tensors without holding one: the batcher prepares the next
/// batch while a session is still busy with the previous one.
#[derive(Debug, Clone, Default)]
pub struct InputSpec(Vec<InputSlot>);

#[derive(Debug, Clone)]
enum InputSlot {
	/// `input_ids`, `attention_mask`, `token_type_ids` or `position_ids`.
	Tokens(String),
	/// `past_key_values.*` of a decoder export: an empty cache for a single pass.
	EmptyKv { name: String, heads: i64, head_dim: i64, fp16: bool },
}

impl InputSpec {
	pub fn of(session: &Session) -> Result<Self> {
		let names: Vec<String> = session.inputs().iter().map(|i| i.name().to_string()).collect();
		let mut slots = Vec::with_capacity(names.len());
		for input in session.inputs() {
			let name = input.name();
			if name.starts_with("past_key_values.") {
				slots.push(empty_kv_slot(input)?);
				continue;
			}
			match name {
				"input_ids" | "attention_mask" | "token_type_ids" | "position_ids" => slots.push(InputSlot::Tokens(name.to_string())),
				other => {
					return Err(Error::Ort(ort::Error::new(format!(
						"model requires unsupported input '{other}'; required inputs: {names:?}"
					))));
				}
			}
		}
		Ok(Self(slots))
	}

	/// The token id/mask/type (and position, empty KV-cache) tensors for `enc`.
	pub fn build(&self, enc: &Encoded) -> Result<SessionInputs<'static, 'static>> {
		let shape = vec![enc.batch as i64, enc.seq as i64];
		let mut map: Vec<(Cow<'static, str>, SessionInputValue<'static>)> = Vec::with_capacity(self.0.len());
		for slot in &self.0 {
			match slot {
				InputSlot::Tokens(name) => {
					let data: Vec<i64> = match name.as_str() {
						"input_ids" => enc.input_ids.concat(),
						"attention_mask" => enc.attention_mask.concat(),
						"token_type_ids" => enc.token_type_ids.concat(),
						// HF convention: position_ids = cumsum(attention_mask) - 1.
						_ => position_ids(&enc.attention_mask),
					};
					map.push((Cow::Owned(name.clone()), Tensor::from_array((shape.clone(), data))?.into()));
				}
				// Decoder-only exports (e.g. Qwen3-Embedding): a single-pass embedding
				// forward runs with an empty KV cache [batch, heads, 0, head_dim].
				InputSlot::EmptyKv { name, heads, head_dim, fp16 } => {
					let kv_shape = vec![enc.batch as i64, *heads, 0, *head_dim];
					let value: SessionInputValue<'static> = if *fp16 {
						Tensor::from_array((kv_shape, Vec::<half::f16>::new()))?.into()
					} else {
						Tensor::from_array((kv_shape, Vec::<f32>::new()))?.into()
					};
					map.push((Cow::Owned(name.clone()), value));
				}
			}
		}
		Ok(SessionInputs::ValueMap(map))
	}
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

/// Spec of an empty KV-cache input `past_key_values.N.{key,value}` of a decoder
/// export: shape [batch, num_kv_heads, 0, head_dim], derived from the declared
/// input shape (dynamic dims are -1 in ORT).
///
/// The export's attention-mask graph only broadcasts correctly with an empty
/// cache (past=0), so a single-pass embedding forward must pass a zero-length
/// cache. Note: CoreML EP rejects zero-element tensors; such models require
/// the CPU EP.
fn empty_kv_slot(input: &Outlet) -> Result<InputSlot> {
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
	let fp16 = match input.dtype().tensor_type() {
		Some(TensorElementType::Float32) => false,
		Some(TensorElementType::Float16) => true,
		Some(other) => {
			return Err(Error::Ort(ort::Error::new(format!(
				"unsupported KV-cache dtype {other:?} for '{}'",
				input.name()
			))));
		}
		None => return Err(Error::Ort(ort::Error::new(format!("input '{}' has no element type", input.name())))),
	};
	Ok(InputSlot::EmptyKv { name: input.name().to_string(), heads: dims[1], head_dim: dims[3], fp16 })
}

#[cfg(test)]
#[path = "tests/tokenize_tests.rs"]
mod tests;
