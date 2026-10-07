//! Unit tests for [`batcher`](super).

use std::{sync::Arc, time::Duration};

use ort::session::Session;

use super::{batch_sizes, share, Batcher};
use crate::{
	config::Batching,
	pipeline::{Extract, RowOut},
	pool::SessionPool,
	tokenize::{Encoded, Row},
	Error, Result,
};

const WAIT: Duration = Duration::from_secs(1);

/// Never started: queued rows stay queued, which is what these tests inspect.
fn batcher(queue_rows: usize) -> Arc<Batcher> {
	let pool = Arc::new(SessionPool::new(Vec::new(), 8));
	let extract: Arc<Extract> = Arc::new(|_: &mut Session, _: &Encoded| -> Result<Vec<RowOut>> { Ok(Vec::new()) });
	Batcher::new(pool, extract, Batching { enabled: true, max_rows: 1, max_tokens: 64, queue_rows })
}

fn rows(n: usize) -> Vec<Row> {
	vec![Row { ids: vec![1], type_ids: vec![0] }; n]
}

#[test]
fn request_is_queued_whole_or_not_at_all() {
	let b = batcher(4);
	let _first = b.enqueue(rows(3), WAIT).unwrap();
	// One slot left: a 2-row request is refused without queuing any of its rows.
	assert!(matches!(b.enqueue(rows(2), WAIT), Err(Error::Saturated)));
	assert_eq!(b.queue_tx.capacity(), 1);
	assert_eq!(b.enqueue(rows(1), WAIT).unwrap().len(), 1);
}

#[test]
fn request_larger_than_queue_is_a_bad_request() {
	let b = batcher(4);
	assert_eq!(b.capacity(), 4);
	assert!(matches!(b.enqueue(rows(5), WAIT), Err(Error::BadRequest(_))));
}

#[test]
fn batches_respect_row_share_and_padded_token_budget() {
	// Share of 4 rows; budget 20 padded tokens: [3,3,4] fits (3 x 4 = 12), a 10
	// would make 4 x 10 = 40; [10,10] is exactly 20.
	assert_eq!(batch_sizes(&[3, 3, 4, 10, 10], 4, 20), [3, 2]);
	assert_eq!(batch_sizes(&[1, 1, 1, 1, 1], 2, 1000), [2, 2, 1]);
	// A row over budget on its own still runs, alone.
	assert_eq!(batch_sizes(&[2, 50], 8, 20), [1, 1]);
	assert!(batch_sizes(&[], 4, 20).is_empty());
}

#[test]
fn batch_errors_keep_their_kind() {
	assert!(matches!(share(&Error::Saturated), Error::Saturated));
	assert!(matches!(share(&Error::PoolTimeout), Error::PoolTimeout));
	assert!(matches!(share(&Error::BadRequest("x".into())), Error::BadRequest(m) if m == "x"));
	assert!(matches!(share(&Error::Tokenize("boom".into())), Error::Ort(_)));
}
