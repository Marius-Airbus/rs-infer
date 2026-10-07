//! Opt-in cross-request dynamic batching for embedding models.
//!
//! Without a batcher each HTTP request tokenizes, pads and forwards its own batch.
//! With `batching:` configured, requests enqueue *rows* (one per text) into a shared
//! queue; a single gatherer task packs rows into batches and hands each batch to one
//! of the per-replica forward workers. Useful when a forward has fixed cost that
//! amortizes across rows (GPU backends, large per-run overhead, heavy padding
//! waste); strictly opt-in because on backends whose cost scales linearly with
//! rows the gain is small and p99 tails grow (a request's rows can span batches).

use std::{
	sync::{
		atomic::{AtomicBool, Ordering},
		Arc, Mutex as StdMutex,
	},
	time::Duration,
};

use tokio::sync::{mpsc, oneshot};

use crate::{
	config::{Batching, Pooling},
	model::OutSel,
	pipeline::{embedding::pool_rows, run_forward},
	pool::SessionPool,
	tokenize::make_token_inputs,
	Error, Result,
};

struct Unit {
	ids: Vec<u32>,
	timeout: Duration,
	reply: oneshot::Sender<Result<Vec<f32>>>,
}

/// Everything a worker needs; derived from `Meta::Embedding` at load time.
pub struct EmbedMeta {
	pub pooling: Pooling,
	pub output: OutSel,
	pub normalize: bool,
	pub dimensions: Option<usize>,
}

pub struct EmbedBatcher {
	queue_tx: mpsc::Sender<Unit>,
	queue_rx: StdMutex<Option<mpsc::Receiver<Unit>>>,
	batch_tx: mpsc::Sender<Vec<Unit>>,
	batch_rx: StdMutex<Option<mpsc::Receiver<Vec<Unit>>>>,
	pool: Arc<SessionPool>,
	meta: Arc<EmbedMeta>,
	settings: Batching,
	started: AtomicBool,
}

impl EmbedBatcher {
	/// Creates channels but spawns nothing; call [`EmbedBatcher::start`] from within
	/// the tokio runtime (models are loaded on a blocking thread otherwise).
	pub(crate) fn new(pool: Arc<SessionPool>, meta: EmbedMeta, settings: Batching) -> Arc<Self> {
		let queue = settings.queue_rows.max(settings.max_rows).max(1);
		let (queue_tx, queue_rx) = mpsc::channel(queue);
		let (batch_tx, batch_rx) = mpsc::channel(pool.replicas().max(1));
		Arc::new(Self {
			queue_tx,
			queue_rx: StdMutex::new(Some(queue_rx)),
			batch_tx,
			batch_rx: StdMutex::new(Some(batch_rx)),
			pool,
			meta: Arc::new(meta),
			settings,
			started: AtomicBool::new(false),
		})
	}

	/// Spawns the gatherer plus one forward worker per session replica. Idempotent.
	pub fn start(self: &Arc<Self>) {
		if self.started.swap(true, Ordering::SeqCst) {
			return;
		}
		let Some(queue_rx) = self.queue_rx.lock().expect("batcher poisoned").take() else {
			return;
		};
		let Some(batch_rx) = self.batch_rx.lock().expect("batcher poisoned").take() else {
			return;
		};
		tokio::spawn(gatherer(Arc::clone(self), queue_rx));
		let batch_rx = Arc::new(tokio::sync::Mutex::new(batch_rx));
		for _ in 0..self.pool.replicas() {
			tokio::spawn(forwarder(Arc::clone(self), batch_rx.clone()));
		}
	}

	/// Submits token rows (one per text, unpadded) and awaits one vector per row,
	/// in order. Fails with Saturated when the queue is full.
	pub(crate) async fn submit(&self, rows: Vec<Vec<u32>>, timeout: Duration) -> Result<Vec<Vec<f32>>> {
		let rxs = self.enqueue(rows, timeout)?;
		let mut out = Vec::with_capacity(rxs.len());
		for reply in rxs {
			out.push(await_reply(reply).await?);
		}
		Ok(out)
	}

	/// Queues all rows of one request or none of them: a request that only half
	/// fits would burn forwards on rows whose caller already got a 429.
	fn enqueue(&self, rows: Vec<Vec<u32>>, timeout: Duration) -> Result<Vec<oneshot::Receiver<Result<Vec<f32>>>>> {
		if rows.is_empty() {
			return Ok(Vec::new());
		}
		let capacity = self.queue_tx.max_capacity();
		if rows.len() > capacity {
			return Err(Error::BadRequest(format!(
				"request has {} inputs but the batching queue holds at most {capacity} (batching.queue_rows)",
				rows.len()
			)));
		}
		let permits = self.queue_tx.try_reserve_many(rows.len()).map_err(|_| Error::Saturated)?;
		Ok(permits
			.zip(rows)
			.map(|(permit, ids)| {
				let (reply_tx, reply_rx) = oneshot::channel();
				permit.send(Unit { ids, timeout, reply: reply_tx });
				reply_rx
			})
			.collect())
	}
}

/// Same error for every caller of a failed batch: `Error` is not `Clone`, but the
/// kinds that map to distinct HTTP statuses (429/503/4xx) must survive the fan-out.
fn share(e: &Error) -> Error {
	match e {
		Error::Saturated => Error::Saturated,
		Error::PoolTimeout => Error::PoolTimeout,
		Error::BadRequest(m) => Error::BadRequest(m.clone()),
		Error::BadOutputShape(s) => Error::BadOutputShape(s.clone()),
		other => Error::Ort(ort::Error::new(other.to_string())),
	}
}

async fn await_reply(reply: oneshot::Receiver<Result<Vec<f32>>>) -> Result<Vec<f32>> {
	match reply.await {
		Ok(inner) => inner,
		Err(_) => Err(Error::Saturated),
	}
}

async fn gatherer(batcher: Arc<EmbedBatcher>, mut queue_rx: mpsc::Receiver<Unit>) {
	let s = batcher.settings;
	while let Some(first) = queue_rx.recv().await {
		// Drain the current backlog (no waiting: rows that don't exist yet
		// can't be predicted), up to what the free sessions can run at once.
		let idle = batcher.pool.available().max(1);
		let staged_cap = s.max_rows.saturating_mul(idle);
		let mut staged = vec![first];
		while staged.len() < staged_cap {
			match queue_rx.try_recv() {
				Ok(u) => staged.push(u),
				Err(_) => break,
			}
		}
		// Length bucketing: rows sorted by token count group similar lengths per
		// batch, so batch padding (to the batch max) is minimized.
		staged.sort_by_key(|u| u.ids.len());
		let share = staged.len().div_ceil(idle).clamp(1, s.max_rows);
		let mut chunks: Vec<Vec<Unit>> = Vec::with_capacity(idle);
		let mut cur: Vec<Unit> = Vec::new();
		let mut cur_tokens = 0usize;
		for u in staged {
			if (cur.len() >= share || cur_tokens + u.ids.len() > s.max_tokens) && !cur.is_empty() {
				cur_tokens = 0;
				chunks.push(std::mem::take(&mut cur));
			}
			cur_tokens += u.ids.len();
			cur.push(u);
		}
		if !cur.is_empty() {
			chunks.push(cur);
		}
		for chunk in chunks {
			if batcher.batch_tx.send(chunk).await.is_err() {
				return;
			}
		}
	}
}

async fn forwarder(batcher: Arc<EmbedBatcher>, batch_rx: Arc<tokio::sync::Mutex<mpsc::Receiver<Vec<Unit>>>>) {
	loop {
		// Lock held only around recv() so waiting for a batch never blocks other
		// workers' forwards.
		let next = {
			let mut rx = batch_rx.lock().await;
			rx.recv().await
		};
		match next {
			Some(units) => run_batch(&batcher, units).await,
			None => return,
		}
	}
}

async fn run_batch(batcher: &Arc<EmbedBatcher>, units: Vec<Unit>) {
	let timeout = units.iter().map(|u| u.timeout).min().unwrap_or(Duration::from_secs(30));
	let tokens = units.iter().map(|u| u.ids.len()).sum::<usize>();
	let (senders, rows): (Vec<_>, Vec<_>) = units.into_iter().map(|u| (u.reply, u.ids)).unzip();
	let outcome = match batcher.pool.acquire(timeout).await {
		Ok(pooled) => {
			let meta = Arc::clone(&batcher.meta);
			pooled
				.run_blocking(move |session| -> Result<Vec<Vec<f32>>> {
					let inputs = make_token_inputs(session, &rows)?;
					let seq = rows.iter().map(|r| r.len()).max().unwrap_or(0);
					let attn: Vec<Vec<i64>> = rows
						.iter()
						.map(|r| {
							let mut v = vec![1i64; r.len()];
							v.resize(seq, 0);
							v
						})
						.collect();
					let mut out = run_forward(session, inputs, &meta.output, |fwd| pool_rows(&fwd, meta.pooling, &attn))?;
					for vec in &mut out {
						if let Some(d) = meta.dimensions {
							if d < vec.len() {
								vec.truncate(d);
							}
						}
						if meta.normalize {
							crate::pipeline::embedding::l2_normalize(vec);
						}
					}
					Ok(out)
				})
				.await
		}
		Err(e) => Err(e),
	};
	match outcome {
		Ok(vectors) => {
			tracing::trace!(rows = vectors.len(), tokens, "dynamic batch done");
			for (sender, vector) in senders.into_iter().zip(vectors) {
				let _ = sender.send(Ok(vector));
			}
		}
		Err(e) => {
			for sender in senders {
				let _ = sender.send(Err(share(&e)));
			}
		}
	}
}

#[cfg(test)]
#[path = "tests/batcher_tests.rs"]
mod tests;
