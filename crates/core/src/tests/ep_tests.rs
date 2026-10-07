//! Unit tests for [`ep`](super).

use super::init_shared_thread_pool;

#[test]
fn shared_thread_pool_is_installed_once() {
	assert_eq!(init_shared_thread_pool(3).unwrap(), 3);
	// Later calls keep the installed pool rather than failing or resizing it.
	assert_eq!(init_shared_thread_pool(5).unwrap(), 3);
}
