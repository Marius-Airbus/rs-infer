//! Unit tests for [`batcher`](super).

use std::{sync::Arc, time::Duration};

use super::{share, EmbedBatcher, EmbedMeta};
use crate::{
	config::{Batching, Pooling},
	model::OutSel,
	pool::SessionPool,
	Error,
};

const WAIT: Duration = Duration::from_secs(1);

/// Not started: queued rows stay queued, which is what these tests inspect.
fn batcher(queue_rows: usize) -> Arc<EmbedBatcher> {
	let pool = Arc::new(SessionPool::new(Vec::new(), 8));
	let meta = EmbedMeta { pooling: Pooling::Mean, output: OutSel("out".into()), normalize: true, dimensions: None };
	EmbedBatcher::new(pool, meta, Batching { max_rows: 1, max_tokens: 64, queue_rows })
}

#[test]
fn request_is_queued_whole_or_not_at_all() {
	let b = batcher(4);
	let _first = b.enqueue(vec![vec![1]; 3], WAIT).unwrap();
	// One slot left: a 2-row request is refused without queuing any of its rows.
	assert!(matches!(b.enqueue(vec![vec![1]; 2], WAIT), Err(Error::Saturated)));
	assert_eq!(b.queue_tx.capacity(), 1);
	assert_eq!(b.enqueue(vec![vec![1]; 1], WAIT).unwrap().len(), 1);
}

#[test]
fn request_larger_than_queue_is_a_bad_request() {
	let b = batcher(4);
	assert!(matches!(b.enqueue(vec![vec![1]; 5], WAIT), Err(Error::BadRequest(_))));
}

#[test]
fn batch_errors_keep_their_kind() {
	assert!(matches!(share(&Error::Saturated), Error::Saturated));
	assert!(matches!(share(&Error::PoolTimeout), Error::PoolTimeout));
	assert!(matches!(share(&Error::BadRequest("x".into())), Error::BadRequest(m) if m == "x"));
	assert!(matches!(share(&Error::Tokenize("boom".into())), Error::Ort(_)));
}
