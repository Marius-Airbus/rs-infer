//! Unit tests for [`api`](super) middleware.

use std::time::Duration;

use axum::{
	body::Body,
	http::{Request, StatusCode},
	routing::{get, post},
	Router,
};
use tower::ServiceExt;

use super::with_middleware;
use crate::metrics::Metrics;

fn app(metrics: &Metrics) -> Router {
	let routes = Router::new()
		.route(
			"/slow",
			get(|| async {
				tokio::time::sleep(Duration::from_secs(5)).await;
				"late"
			}),
		)
		.route("/echo", post(|body: String| async move { body }));
	with_middleware(routes, metrics.clone(), Duration::from_millis(50), 16)
}

#[tokio::test]
async fn timed_out_requests_are_counted() {
	let metrics = Metrics::default();
	let res = app(&metrics).oneshot(Request::get("/slow").body(Body::empty()).unwrap()).await.unwrap();
	assert_eq!(res.status(), StatusCode::REQUEST_TIMEOUT);
	let text = metrics.encode();
	assert!(text.contains(r#"rsinfer_http_requests_total{route="/slow",status="408"} 1"#), "{text}");
}

#[tokio::test]
async fn oversized_bodies_are_counted() {
	let metrics = Metrics::default();
	let req = Request::post("/echo").header("content-length", "64").body(Body::from(vec![b'x'; 64])).unwrap();
	let res = app(&metrics).oneshot(req).await.unwrap();
	assert_eq!(res.status(), StatusCode::PAYLOAD_TOO_LARGE);
	let text = metrics.encode();
	assert!(text.contains(r#"rsinfer_http_requests_total{route="/echo",status="413"} 1"#), "{text}");
}
