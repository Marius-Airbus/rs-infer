//! Unit tests for [`ep`](super).

use super::init_shared_thread_pool;

#[test]
fn shared_thread_pool_is_installed_once() {
	// Other tests building sessions install the pool too; whoever runs first wins.
	let n = init_shared_thread_pool(3).unwrap();
	assert!(n == 3 || n == 2, "unexpected pool size {n}");
	// Later calls keep the installed pool rather than failing or resizing it.
	assert_eq!(init_shared_thread_pool(5).unwrap(), n);
}
