//! Pipelines: turn text/tokens into typed predictions through ORT sessions.

pub mod embedding;
pub mod pii;
pub mod rerank;
pub mod zeroshot;

use std::{sync::Arc, time::Duration};

use ort::session::{OutputSelector, RunOptions, Session, SessionOutputs};

use crate::{
	model::OutSel,
	tokenize::{make_inputs, Encoded},
	LoadedModel,
};

/// One row's forward-pass result, already post-processed for its model kind.
#[derive(Debug, Clone)]
pub enum RowOut {
	/// Embedding (pooled, truncated to `dimensions`, normalized as configured).
	Vector(Vec<f32>),
	/// Rerank relevance score.
	Score(f64),
	/// Raw logits of one sequence-classification row (zero-shot NLI).
	Logits(Vec<f32>),
	/// Per token: (argmax label id, its probability), padding positions included.
	Tokens(Vec<(usize, f64)>),
}

impl RowOut {
	pub(crate) fn into_vector(self) -> crate::Result<Vec<f32>> {
		match self {
			RowOut::Vector(v) => Ok(v),
			other => Err(other.mismatch("vector")),
		}
	}

	pub(crate) fn into_score(self) -> crate::Result<f64> {
		match self {
			RowOut::Score(s) => Ok(s),
			other => Err(other.mismatch("score")),
		}
	}

	pub(crate) fn into_logits(self) -> crate::Result<Vec<f32>> {
		match self {
			RowOut::Logits(l) => Ok(l),
			other => Err(other.mismatch("logits")),
		}
	}

	pub(crate) fn into_tokens(self) -> crate::Result<Vec<(usize, f64)>> {
		match self {
			RowOut::Tokens(t) => Ok(t),
			other => Err(other.mismatch("tokens")),
		}
	}

	fn mismatch(&self, wanted: &str) -> crate::Error {
		let got = match self {
			RowOut::Vector(_) => "vector",
			RowOut::Score(_) => "score",
			RowOut::Logits(_) => "logits",
			RowOut::Tokens(_) => "tokens",
		};
		crate::Error::Ort(ort::Error::new(format!("model produced {got} rows where {wanted} rows were expected")))
	}
}

/// Forward pass + per-row post-processing of one right-padded batch. Built per
/// model at load time and shared by the batcher and the direct path.
pub type Extract = dyn Fn(&mut Session, &Encoded) -> crate::Result<Vec<RowOut>> + Send + Sync;

/// Runs `enc`'s rows through `model`, one output per row in input order: via the
/// cross-request batcher when the model has one and the request fits its queue,
/// else directly on one session in `max_batch`-row slices.
pub(crate) async fn run_rows(model: &Arc<LoadedModel>, enc: Encoded, queue_wait: Duration) -> crate::Result<Vec<RowOut>> {
	if let Some(batcher) = model.batcher.as_ref().filter(|b| enc.batch <= b.capacity()) {
		return batcher.submit(enc.into_rows(), queue_wait).await;
	}
	let extract = Arc::clone(&model.extract);
	let max_rows = model.cfg.max_batch;
	let pooled = model.pool.acquire(queue_wait).await?;
	pooled
		.run_blocking(move |session| {
			let mut out = Vec::with_capacity(enc.batch);
			for part in enc.split(max_rows) {
				out.extend(extract(session, &part)?);
			}
			Ok(out)
		})
		.await
}

/// Builds `enc`'s inputs, runs the model and hands the selected output to `f`.
pub(crate) fn forward<R>(session: &mut Session, enc: &Encoded, sel: &OutSel, f: impl FnOnce(Fwd<'_>) -> crate::Result<R>) -> crate::Result<R> {
	let inputs = make_inputs(session, enc)?;
	run_forward(session, inputs, sel, f)
}

pub(crate) struct Fwd<'a> {
	pub shape: Vec<usize>,
	pub data: &'a [f32],
}

/// Runs a CPU-bound closure (tokenization) on the blocking pool so async runtime
/// workers stay free for request handling.
pub(crate) async fn blocking<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> crate::Result<T> {
	tokio::task::spawn_blocking(f).await.map_err(|e| crate::Error::Config(format!("blocking task panicked: {e}")))
}

/// Runs a forward pass and hands the selected output to `f` as a borrowed view
/// (shape + contiguous f32 slice) — no copy of the full activation tensor.
/// Decoder exports would otherwise materialize all KV-cache `present.*` outputs.
pub(crate) fn run_forward<R>(
	session: &mut Session,
	inputs: ort::session::SessionInputs<'static, 'static>,
	sel: &OutSel,
	f: impl FnOnce(Fwd<'_>) -> crate::Result<R>,
) -> crate::Result<R> {
	let names: Vec<String> = session.outputs().iter().map(|o| o.name().to_string()).collect();
	let options = RunOptions::new()?.with_outputs(OutputSelector::no_default().with(sel.0.clone()));
	let outputs: SessionOutputs = session.run_with_options(inputs, &options)?;
	let value = outputs
		.get(&sel.0)
		.ok_or_else(|| crate::Error::Ort(ort::Error::new(format!("output '{}' not found; model has {names:?}", sel.0))))?;
	let (shape, data) = value.try_extract_tensor::<f32>()?;
	f(Fwd {
		shape: shape.iter().map(|&d| d as usize).collect(),
		data,
	})
}

/// The tokenizer cuts inputs longer than `max_len`; say so rather than silently
/// embedding or scoring a prefix.
pub(crate) fn warn_truncated(model: &LoadedModel, truncated: usize) {
	if truncated > 0 {
		tracing::warn!(model = model.name(), inputs = truncated, max_len = ?model.max_len, "inputs longer than max_len were truncated");
	}
}

pub(crate) fn softmax(logits: &[f32]) -> Vec<f64> {
	let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
	let exps: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
	let sum: f32 = exps.iter().sum();
	exps.iter().map(|e| (e / sum) as f64).collect()
}
