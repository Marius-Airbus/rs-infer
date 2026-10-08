//! Cross-request dynamic batching, on by default for every model kind.
//!
//! Requests enqueue *rows* (one per text or text pair, unpadded) into a shared
//! queue. A gatherer task drains the backlog, sorts it by length (so a batch
//! pads little) and cuts it into batches bounded by `max_rows` and a padded-token
//! budget (`max_tokens` >= rows x longest row, which also bounds activation
//! memory), spread over the idle session replicas. A row arriving to an empty
//! queue is dispatched at once, so batching never waits for company; it pays off
//! when requests overlap: 1-text embedding requests went from 381 to 655 req/s
//! (e5-small int8, 14-core i7, 4 replicas).

use std::{
	sync::{
		atomic::{AtomicBool, Ordering},
		Arc, Mutex as StdMutex,
	},
	time::Duration,
};

use tokio::sync::{mpsc, oneshot};

use crate::{
	config::Batching,
	pipeline::{Extract, RowOut},
	pool::SessionPool,
	tokenize::{Encoded, InputSpec, Row},
	Error, Result,
};

struct Unit {
	row: Row,
	timeout: Duration,
	reply: oneshot::Sender<Result<RowOut>>,
}

pub struct Batcher {
	queue_tx: mpsc::Sender<Unit>,
	queue_rx: StdMutex<Option<mpsc::Receiver<Unit>>>,
	batch_tx: mpsc::Sender<Vec<Unit>>,
	batch_rx: StdMutex<Option<mpsc::Receiver<Vec<Unit>>>>,
	pool: Arc<SessionPool>,
	extract: Arc<Extract>,
	inputs: Arc<InputSpec>,
	settings: Batching,
	/// Forward workers per session replica (see [`Batcher::start`]).
	workers_per_replica: usize,
	started: AtomicBool,
}

impl Batcher {
	/// Creates the channels; tasks are spawned on first use (models load on
	/// blocking threads, outside the runtime).
	pub(crate) fn new(pool: Arc<SessionPool>, extract: Arc<Extract>, inputs: Arc<InputSpec>, settings: Batching, workers_per_replica: usize) -> Arc<Self> {
		let queue = settings.queue_rows.max(settings.max_rows).max(1);
		let (queue_tx, queue_rx) = mpsc::channel(queue);
		let (batch_tx, batch_rx) = mpsc::channel(pool.replicas().max(1));
		Arc::new(Self {
			queue_tx,
			queue_rx: StdMutex::new(Some(queue_rx)),
			batch_tx,
			batch_rx: StdMutex::new(Some(batch_rx)),
			pool,
			extract,
			inputs,
			settings,
			workers_per_replica: workers_per_replica.max(1),
			started: AtomicBool::new(false),
		})
	}

	/// Most rows one request can queue; bigger requests must run directly.
	pub fn capacity(&self) -> usize {
		self.queue_tx.max_capacity()
	}

	/// Spawns the gatherer plus `workers_per_replica` forward workers per session
	/// replica. With two (accelerators), one worker prepares the next batch's
	/// input tensors while the other's batch runs, so the device never waits for
	/// that; on CPU one per replica, as the CPU is busy with the forward anyway
	/// and more workers would only split the backlog into smaller batches. Idempotent.
	fn start(self: &Arc<Self>) {
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
		for _ in 0..self.pool.replicas() * self.workers_per_replica {
			tokio::spawn(forwarder(Arc::clone(self), batch_rx.clone()));
		}
	}

	/// Submits rows and awaits one output per row, in order. Fails with
	/// Saturated when the queue is full.
	pub(crate) async fn submit(self: &Arc<Self>, rows: Vec<Row>, timeout: Duration) -> Result<Vec<RowOut>> {
		self.start();
		let rxs = self.enqueue(rows, timeout)?;
		let mut out = Vec::with_capacity(rxs.len());
		for reply in rxs {
			out.push(await_reply(reply).await?);
		}
		Ok(out)
	}

	/// Queues all rows of one request or none of them: a request that only half
	/// fits would burn forwards on rows whose caller already got a 429.
	fn enqueue(&self, rows: Vec<Row>, timeout: Duration) -> Result<Vec<oneshot::Receiver<Result<RowOut>>>> {
		if rows.is_empty() {
			return Ok(Vec::new());
		}
		let capacity = self.capacity();
		if rows.len() > capacity {
			return Err(Error::BadRequest(format!(
				"request has {} inputs but the batching queue holds at most {capacity} (batching.queue_rows)",
				rows.len()
			)));
		}
		let permits = self.queue_tx.try_reserve_many(rows.len()).map_err(|_| Error::Saturated)?;
		Ok(permits
			.zip(rows)
			.map(|(permit, row)| {
				let (reply_tx, reply_rx) = oneshot::channel();
				permit.send(Unit { row, timeout, reply: reply_tx });
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

async fn await_reply(reply: oneshot::Receiver<Result<RowOut>>) -> Result<RowOut> {
	match reply.await {
		Ok(inner) => inner,
		Err(_) => Err(Error::Saturated),
	}
}

/// Cuts rows sorted by length into consecutive batches and returns their sizes.
/// A batch closes at `share` rows, or before a row that would push its padded
/// size (rows x longest row, i.e. that row) past `max_tokens`; a single row
/// always forms a batch.
pub(crate) fn batch_sizes(sorted_lens: &[usize], share: usize, max_tokens: usize) -> Vec<usize> {
	let mut sizes = Vec::new();
	let mut cur = 0usize;
	for &len in sorted_lens {
		if cur > 0 && (cur >= share || (cur + 1) * len > max_tokens) {
			sizes.push(cur);
			cur = 0;
		}
		cur += 1;
	}
	if cur > 0 {
		sizes.push(cur);
	}
	sizes
}

async fn gatherer(batcher: Arc<Batcher>, mut queue_rx: mpsc::Receiver<Unit>) {
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
		staged.sort_by_key(|u| u.row.ids.len());
		let share = staged.len().div_ceil(idle).clamp(1, s.max_rows);
		let lens: Vec<usize> = staged.iter().map(|u| u.row.ids.len()).collect();
		let mut staged = staged.into_iter();
		for size in batch_sizes(&lens, share, s.max_tokens) {
			let batch: Vec<Unit> = staged.by_ref().take(size).collect();
			if batcher.batch_tx.send(batch).await.is_err() {
				return;
			}
		}
	}
}

async fn forwarder(batcher: Arc<Batcher>, batch_rx: Arc<tokio::sync::Mutex<mpsc::Receiver<Vec<Unit>>>>) {
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

async fn run_batch(batcher: &Arc<Batcher>, units: Vec<Unit>) {
	let timeout = units.iter().map(|u| u.timeout).min().unwrap_or(Duration::from_secs(30));
	let (senders, rows): (Vec<_>, Vec<_>) = units.into_iter().map(|u| (u.reply, u.row)).unzip();
	let n = rows.len();
	// Input tensors are built before a session is taken, off the session's time.
	let spec = Arc::clone(&batcher.inputs);
	let prepared = crate::pipeline::blocking(move || {
		let enc = Encoded::from_rows(&rows);
		spec.build(&enc).map(|inputs| (enc, inputs))
	})
	.await
	.and_then(|r| r);
	let outcome = match prepared {
		Ok((enc, inputs)) => match batcher.pool.acquire(timeout).await {
			Ok(pooled) => {
				let extract = Arc::clone(&batcher.extract);
				pooled.run_blocking(move |session| extract(session, inputs, &enc)).await
			}
			Err(e) => Err(e),
		},
		Err(e) => Err(e),
	};
	let outcome = outcome.and_then(|outs| {
		if outs.len() == n {
			Ok(outs)
		} else {
			Err(Error::Ort(ort::Error::new(format!("batch of {n} rows produced {} outputs", outs.len()))))
		}
	});
	match outcome {
		Ok(outs) => {
			tracing::trace!(rows = n, "dynamic batch done");
			for (sender, out) in senders.into_iter().zip(outs) {
				let _ = sender.send(Ok(out));
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
