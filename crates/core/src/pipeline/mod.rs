//! Pipelines: turn text/tokens into typed predictions through ORT sessions.

pub mod embedding;
pub mod pii;
pub mod rerank;
pub mod zeroshot;

use ort::session::{OutputSelector, RunOptions, Session, SessionOutputs};

use crate::{
	model::OutSel,
	tokenize::{make_inputs, Encoded},
};

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

/// Forwards `enc` in slices of at most `max_rows` rows so one big request cannot
/// blow up activation memory; `f` maps each slice's output to one item per row
/// (it also gets the slice, e.g. for its attention mask).
pub(crate) fn forward_rows<R>(
	session: &mut Session,
	enc: &Encoded,
	max_rows: usize,
	sel: &OutSel,
	mut f: impl FnMut(Fwd<'_>, &Encoded) -> crate::Result<Vec<R>>,
) -> crate::Result<Vec<R>> {
	let mut out = Vec::with_capacity(enc.batch);
	for part in enc.split(max_rows) {
		let inputs = make_inputs(session, &part)?;
		out.extend(run_forward(session, inputs, sel, |fwd| f(fwd, &part))?);
	}
	Ok(out)
}

pub(crate) fn softmax(logits: &[f32]) -> Vec<f64> {
	let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
	let exps: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
	let sum: f32 = exps.iter().sum();
	exps.iter().map(|e| (e / sum) as f64).collect()
}
