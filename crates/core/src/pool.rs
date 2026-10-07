use std::{
	sync::{
		atomic::{AtomicUsize, Ordering},
		Arc, Mutex as StdMutex,
	},
	time::Duration,
};

use ort::session::Session;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::{Error, Result};

/// A pool of independent ORT sessions for one model.
///
/// `ort::session::Session` is `Send` but not `Sync` (and `run` takes `&mut self`),
/// so concurrency comes from replicas, matching ONNX Runtime's own guidance.
/// Generic only so tests can pool plain values; the server always pools `Session`s.
pub struct SessionPool<S = Session> {
	sessions: Arc<StdMutex<Vec<S>>>,
	sem: Arc<Semaphore>,
	replicas: usize,
	max_queue: usize,
	waiting: AtomicUsize,
}

impl<S> SessionPool<S> {
	pub fn new(sessions: Vec<S>, max_queue: usize) -> Self {
		let replicas = sessions.len();
		Self {
			sessions: Arc::new(StdMutex::new(sessions)),
			sem: Arc::new(Semaphore::new(replicas)),
			replicas,
			max_queue,
			waiting: AtomicUsize::new(0),
		}
	}

	pub fn replicas(&self) -> usize {
		self.replicas
	}

	/// Sessions not currently checked out (idle or gathered-but-not-started).
	pub fn available(&self) -> usize {
		self.sem.available_permits()
	}

	pub async fn acquire(&self, timeout: Duration) -> Result<Pooled<S>> {
		if self.sem.available_permits() == 0 && self.waiting.load(Ordering::Relaxed) >= self.max_queue {
			return Err(Error::Saturated);
		}
		self.waiting.fetch_add(1, Ordering::SeqCst);
		struct Guard<'a>(&'a AtomicUsize);
		impl Drop for Guard<'_> {
			fn drop(&mut self) {
				self.0.fetch_sub(1, Ordering::SeqCst);
			}
		}
		let guard = Guard(&self.waiting);
		let permit = tokio::time::timeout(timeout, Arc::clone(&self.sem).acquire_owned())
			.await
			.map_err(|_| Error::PoolTimeout)?
			.map_err(|_| Error::Saturated)?;
		let session = self
			.sessions
			.lock()
			.expect("session pool poisoned")
			.pop()
			.ok_or(Error::Saturated)?;
		drop(guard);
		Ok(Pooled {
			session: Some(session),
			sessions: Arc::clone(&self.sessions),
			_permit: permit,
		})
	}
}

/// A checked-out session. Dropping it puts the session back, then releases the permit.
pub struct Pooled<S = Session> {
	session: Option<S>,
	sessions: Arc<StdMutex<Vec<S>>>,
	_permit: OwnedSemaphorePermit,
}

impl<S> Pooled<S> {
	pub fn get_mut(&mut self) -> &mut S {
		self.session.as_mut().expect("session taken")
	}

	/// Runs a blocking inference closure with the session off the async runtime.
	///
	/// The blocking task owns the checkout, so the session returns to the pool when
	/// the work ends, even if `f` panics or this future is dropped mid-inference
	/// (request timeout, client disconnect). Otherwise the permit would come back
	/// without its session and later acquires would fail with `Saturated`.
	pub async fn run_blocking<T, F>(mut self, f: F) -> Result<T>
	where
		S: Send + 'static,
		T: Send + 'static,
		F: FnOnce(&mut S) -> Result<T> + Send + 'static,
	{
		tokio::task::spawn_blocking(move || f(self.get_mut()))
			.await
			.map_err(|e| Error::Ort(ort::Error::new(format!("inference task panicked: {e}"))))?
	}
}

impl<S> Drop for Pooled<S> {
	fn drop(&mut self) {
		if let Some(session) = self.session.take() {
			if let Ok(mut sessions) = self.sessions.lock() {
				sessions.push(session);
			}
		}
	}
}

#[cfg(test)]
#[path = "tests/pool_tests.rs"]
mod tests;
