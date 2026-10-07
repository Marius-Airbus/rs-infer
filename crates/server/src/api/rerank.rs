use axum::{extract::State, Json};
use rsinfer_core::{
	Kind,
	pipeline::rerank,
};

use super::usage;
use crate::{
	dto::{RerankDocument, RerankRequest, RerankResponse, RerankResult, ScoreItem, ScoreRequest, ScoreResponse},
	error::ApiError,
	state::AppState,
};

/// vLLM/Jina-compatible rerank endpoint.
pub async fn rerank_h(State(state): State<AppState>, Json(req): Json<RerankRequest>) -> Result<Json<RerankResponse>, ApiError> {
	let model = state.registry.resolve(req.model.as_deref(), Kind::Rerank)?;
	if req.documents.is_empty() {
		return Err(ApiError(rsinfer_core::Error::BadRequest("`documents` is empty".into())));
	}
	let outcome = rerank::score_pairs(&model, req.query, req.documents.clone(), req.max_chunks_per_doc, state.queue_wait).await?;
	let return_docs = req.return_documents.unwrap_or(true);

	let mut scored = outcome.scores;
	scored.sort_by(|a, b| b.score.total_cmp(&a.score));
	if let Some(n) = req.top_n {
		scored.truncate(n);
	}
	let results = scored
		.into_iter()
		.map(|s| RerankResult {
			index: s.index,
			document: return_docs.then(|| RerankDocument { text: req.documents[s.index].clone() }),
			relevance_score: s.score,
		})
		.collect();
	Ok(Json(RerankResponse {
		id: uuid::Uuid::new_v4().to_string(),
		model: model.name().to_string(),
		results,
		usage: usage(outcome.prompt_tokens),
	}))
}

/// vLLM-compatible /v1/score: cross-encoder scoring of the cross-product of text_1 x text_2.
pub async fn score_h(State(state): State<AppState>, Json(req): Json<ScoreRequest>) -> Result<Json<ScoreResponse>, ApiError> {
	let model = state.registry.resolve(req.model.as_deref(), Kind::Rerank)?;
	let a = req.text_1.into_vec();
	let b = req.text_2.into_vec();
	if a.is_empty() || b.is_empty() {
		return Err(ApiError(rsinfer_core::Error::BadRequest("text_1/text_2 must not be empty".into())));
	}
	let pairs: Vec<(String, String)> = a.iter().flat_map(|x| b.iter().map(move |y| (x.clone(), y.clone()))).collect();
	let outcome = rerank::score_text_pairs(&model, pairs.clone(), state.queue_wait).await?;
	let data = outcome
		.scores
		.iter()
		.map(|s| ScoreItem {
			object: "score",
			index: s.index,
			text_1: pairs[s.index].0.clone(),
			text_2: pairs[s.index].1.clone(),
			score: s.score,
		})
		.collect();
	Ok(Json(ScoreResponse {
		object: "list",
		data,
		model: model.name().to_string(),
		usage: usage(outcome.prompt_tokens),
	}))
}
