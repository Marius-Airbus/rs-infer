mod classify;
mod embeddings;
mod pii;
mod rerank;

use std::time::{Duration, Instant};

use axum::{
	body::Body,
	extract::{MatchedPath, Request, State},
	http::StatusCode,
	middleware::{self, Next},
	response::{IntoResponse, Response},
	routing::{get, post},
	Json, Router,
};
use tower_http::{limit::RequestBodyLimitLayer, timeout::TimeoutLayer};

use crate::{metrics::Metrics, state::AppState};

pub fn router(state: AppState, request_timeout: Duration, max_body_bytes: usize) -> Router {
	let metrics = state.metrics.clone();
	let routes = Router::new()
		.route("/health", get(health).post(health))
		.route("/v1/models", get(list_models))
		.route("/v1/embeddings", post(embeddings::embeddings))
		.route("/embed", post(embeddings::embed_tei))
		.route("/v1/rerank", post(rerank::rerank_h))
		.route("/rerank", post(rerank::rerank_h))
		.route("/v1/score", post(rerank::score_h))
		.route("/classify/zero-shot", post(classify::classify_h))
		.route("/v1/classify/zero-shot", post(classify::classify_h))
		.route("/classify/true-false", post(classify::true_false_h))
		.route("/v1/classify/true-false", post(classify::true_false_h))
		.route("/pii/detect", post(pii::pii_detect_h))
		.route("/v1/pii/detect", post(pii::pii_detect_h))
		.route("/pii/redact", post(pii::pii_redact_h))
		.route("/v1/pii/redact", post(pii::pii_redact_h))
		.route("/metrics", get(metrics_h))
		.with_state(state);
	with_middleware(routes, metrics, request_timeout, max_body_bytes)
}

/// `track` is the outermost layer so requests cut short by the timeout (408) or
/// the body limit (413) are still logged and counted in /metrics.
fn with_middleware(routes: Router, metrics: Metrics, request_timeout: Duration, max_body_bytes: usize) -> Router {
	routes
		.layer(TimeoutLayer::with_status_code(StatusCode::REQUEST_TIMEOUT, request_timeout))
		.layer(RequestBodyLimitLayer::new(max_body_bytes))
		.layer(middleware::from_fn_with_state(metrics, track))
}

/// Shared usage object: our pipelines only consume prompt tokens.
pub(crate) fn usage(tokens: usize) -> crate::dto::Usage {
	crate::dto::Usage { prompt_tokens: tokens, total_tokens: tokens }
}

async fn track(State(metrics): State<Metrics>, req: Request<Body>, next: Next) -> Response {
	let start = Instant::now();
	let method = req.method().clone();
	let route = req
		.extensions()
		.get::<MatchedPath>()
		.map(|p| p.as_str().to_string())
		.unwrap_or_else(|| req.uri().path().to_string());
	let res = next.run(req).await;
	let status = res.status().as_u16();
	let elapsed = start.elapsed();
	metrics.observe(&route, status, elapsed);
	tracing::info!(
		method = %method,
		route = %route,
		status,
		duration_ms = elapsed.as_millis(),
		"http request"
	);
	res
}

async fn health() -> impl IntoResponse {
	Json(serde_json::json!({ "status": "ok" }))
}

#[derive(serde::Serialize)]
struct ModelCard {
	id: String,
	object: &'static str,
	kind: &'static str,
	source: String,
	execution_providers: Vec<String>,
	replicas: usize,
	max_len: Option<usize>,
	default: bool,
}

async fn list_models(State(state): State<AppState>) -> Json<serde_json::Value> {
	let data: Vec<ModelCard> = state
		.registry
		.infos()
		.into_iter()
		.map(|i| ModelCard {
			id: i.name,
			object: "model",
			kind: i.kind,
			source: i.source,
			execution_providers: i.execution_providers,
			replicas: i.replicas,
			max_len: i.max_len,
			default: i.default,
		})
		.collect();
	Json(serde_json::json!({ "object": "list", "data": data }))
}

async fn metrics_h(State(state): State<AppState>) -> impl IntoResponse {
	([(axum::http::header::CONTENT_TYPE, "text/plain; version=0.0.4; charset=utf-8")], state.metrics.encode())
}

#[cfg(test)]
#[path = "../tests/api_tests.rs"]
mod tests;
