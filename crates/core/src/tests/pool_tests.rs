//! Unit tests for [`pool`](super).

use std::time::Duration;

use super::SessionPool;
use crate::Result;

const WAIT: Duration = Duration::from_secs(5);

#[tokio::test]
async fn session_returns_to_pool_after_panic() {
	let pool = SessionPool::new(vec![7u32], 8);
	let out = pool.acquire(WAIT).await.unwrap().run_blocking(|_| -> Result<()> { panic!("boom") }).await;
	assert!(out.is_err());
	// With a single replica, a lost session would make every later acquire fail.
	let out = pool.acquire(WAIT).await.unwrap().run_blocking(|s| Ok(*s)).await;
	assert_eq!(out.unwrap(), 7);
}

#[tokio::test]
async fn session_returns_to_pool_when_caller_gives_up() {
	let pool = SessionPool::new(vec![7u32], 8);
	let slow = pool.acquire(WAIT).await.unwrap().run_blocking(|_| {
		std::thread::sleep(Duration::from_millis(200));
		Ok(())
	});
	// Request timeout / client disconnect: the future is dropped mid-inference.
	assert!(tokio::time::timeout(Duration::from_millis(20), slow).await.is_err());
	// The slot frees up once the inference finishes, with its session.
	let out = pool.acquire(WAIT).await.unwrap().run_blocking(|s| Ok(*s)).await;
	assert_eq!(out.unwrap(), 7);
}
