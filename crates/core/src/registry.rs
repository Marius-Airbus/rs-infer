use std::{
	collections::HashMap,
	path::{Path, PathBuf},
	sync::Arc,
};

use ort::session::Session;

use crate::{
	batcher::Batcher,
	config::{Config, Dtype, Kind, ModelConfig, Pooling, Scoring},
	ep::{new_session, resolve_eps},
	graph_pooling, hub,
	model::{find_label, id2label_from_config, pick_output, pooling_from_st_config, Meta},
	pipeline::{self, Extract},
	pool::SessionPool,
	quantize,
	tokenize::{Encoder, InputSpec},
	Error, LoadedModel, Result,
};

#[derive(Debug, Clone, serde::Serialize)]
pub struct ModelInfo {
	pub name: String,
	pub kind: &'static str,
	pub source: String,
	pub execution_providers: Vec<String>,
	pub replicas: usize,
	pub max_len: Option<usize>,
	pub default: bool,
}

pub struct Registry {
	models: HashMap<String, Arc<LoadedModel>>,
	defaults: HashMap<Kind, String>,
}

impl Registry {
	/// Downloads/resolves all models and builds session pools in parallel.
	pub async fn load(config: Arc<Config>) -> Result<Self> {
		let mut handles = Vec::new();
		for model in &config.models {
			let cfg = model.clone();
			let cache = config.server.hf_cache_dir.clone();
			let max_queue = config.server.max_queue;
			handles.push(tokio::task::spawn_blocking(move || load_model(&cfg, cache.as_deref(), max_queue)));
		}
		let mut models = HashMap::new();
		let mut by_kind: HashMap<Kind, Vec<String>> = HashMap::new();
		// (by_kind is consumed to compute per-kind defaults below)
		let mut errors: Vec<String> = Vec::new();
		for handle in handles {
			match handle.await.map_err(|e| Error::Config(format!("loader task panicked: {e}")))? {
				Ok(model) => {
					let name = model.name().to_string();
					let kind = model.kind();
					by_kind.entry(kind).or_default().push(name.clone());
					models.insert(name, model);
				}
				Err(e) => errors.push(format!("{e}")),
			}
		}
		if !errors.is_empty() {
			return Err(Error::Config(format!(
				"failed to load {} model(s):\n{}",
				errors.len(),
				errors.join("\n")
			)));
		}

		let mut defaults: HashMap<Kind, String> = HashMap::new();
		for kind in [Kind::Embedding, Kind::Rerank, Kind::Zeroshot, Kind::Pii] {
			let Some(names) = by_kind.get(&kind) else { continue };
			let explicit = names.iter().find(|n| models.get(*n).is_some_and(|m| m.cfg.default));
			let default = explicit.or_else(|| (names.len() == 1).then(|| &names[0])).cloned();
			if let Some(d) = default {
				defaults.insert(kind, d);
			}
		}
		let _ = &by_kind;
		Ok(Self { models, defaults })
	}

	pub fn resolve(&self, requested: Option<&str>, kind: Kind) -> Result<Arc<LoadedModel>> {
		let name = match requested {
			Some(n) => n.to_string(),
			None => self
				.defaults
				.get(&kind)
				.cloned()
				.ok_or_else(|| Error::BadRequest(format!("no default {} model configured; specify `model`", kind.as_str())))?,
		};
		let model = self.models.get(&name).ok_or(Error::ModelNotFound(name.clone()))?;
		if model.kind() != kind {
			return Err(Error::KindMismatch {
				name,
				expected: kind.as_str(),
				actual: model.kind().as_str(),
			});
		}
		Ok(model.clone())
	}

	pub fn infos(&self) -> Vec<ModelInfo> {
		let mut out: Vec<ModelInfo> = self
			.models
			.values()
			.map(|m| ModelInfo {
				name: m.name().to_string(),
				kind: m.kind().as_str(),
				source: m.source.clone(),
				execution_providers: m.eps.clone(),
				replicas: m.pool.replicas(),
				max_len: m.max_len,
				default: self.defaults.get(&m.kind()).map(|d| d == m.name()).unwrap_or(false),
			})
			.collect();
		out.sort_by(|a, b| a.name.cmp(&b.name));
		out
	}
}

/// Loads one model (download/resolve, tokenizer, session pool, metadata).
/// Used by [`Registry::load`] and the server's `profile` command.
pub fn load_model(cfg: &ModelConfig, hf_cache: Option<&std::path::Path>, max_queue: usize) -> Result<Arc<LoadedModel>> {
	cfg.validate()?;
	let t_start = std::time::Instant::now();
	let rss_before = crate::memory::rss_mb().unwrap_or(0);
	tracing::info!(model = cfg.name.as_str(), kind = cfg.kind.as_str(), replicas = cfg.replicas, "loading model");

	let (eps, _) = resolve_eps(cfg);
	let resolved = hub::resolve(&graph_preference(cfg, &eps), hf_cache)?;
	let model_mb = crate::memory::model_size_mb(&resolved.model).unwrap_or(0);
	tracing::info!(
		model = cfg.name.as_str(),
		source = %resolved.source,
		model_file = %resolved.model.display(),
		model_mb,
		elapsed_ms = t_start.elapsed().as_millis(),
		"model files resolved (incl. download if not cached)"
	);
	// PII scans long texts in windows overlapping by a quarter of max_len (context
	// for tokens near a window edge); other kinds just truncate.
	let stride = match (cfg.kind, cfg.max_len) {
		(Kind::Pii, Some(max)) if max >= 16 => max / 4,
		_ => 0,
	};
	let encoder = Encoder::new(&resolved.tokenizer, cfg.max_len, stride)?;

	let (graph, precision) = choose_graph(cfg, &resolved, &encoder, &eps, hf_cache);
	let (graph, pooling_in_graph) = pool_in_graph(cfg, &resolved, graph, &eps, hf_cache);
	let t_sessions = std::time::Instant::now();
	let mut sessions = Vec::with_capacity(cfg.replicas);
	for _ in 0..cfg.replicas {
		sessions.push(new_session(&graph, cfg)?);
	}
	let session_ms = t_sessions.elapsed().as_millis();
	let first = sessions.first().expect("replicas >= 1, validated");
	let input_names = first.inputs().iter().map(|i| i.name().to_string()).collect();
	let output_names = first.outputs().iter().map(|o| o.name().to_string()).collect();
	let meta = build_meta(cfg, &resolved, &encoder, first)?;
	let inputs = Arc::new(InputSpec::of(first)?);

	let rss_after = crate::memory::rss_mb().unwrap_or(0);
	let rss_used = rss_after.saturating_sub(rss_before);
	tracing::info!(
		model = cfg.name.as_str(),
		kind = cfg.kind.as_str(),
		source = %resolved.source,
		model_mb,
		replicas = sessions.len(),
		eps = %eps.join(","),
		max_len = %cfg.max_len.map(|v| v.to_string()).unwrap_or_else(|| "unset".into()),
		batching = cfg.batching.enabled,
		precision,
		pooling_in_graph,
		session_build_ms = session_ms,
		elapsed_ms = t_start.elapsed().as_millis(),
		ram_used_mb = rss_used,
		ram_total_mb = rss_after,
		"model loaded"
	);
	let pool = Arc::new(SessionPool::new(sessions, max_queue));
	let extract = build_extract(&meta);
	// Accelerators: prepare the next batch while one runs (2 workers per session).
	let workers_per_replica = if eps.iter().all(|e| e == "cpu") { 1 } else { 2 };
	let batcher = cfg.batching.enabled.then(|| Batcher::new(pool.clone(), extract.clone(), inputs.clone(), cfg.batching, workers_per_replica));
	Ok(Arc::new(LoadedModel {
		cfg: cfg.clone(),
		encoder,
		pool,
		extract,
		inputs,
		batcher,
		meta,
		source: resolved.source,
		eps,
		max_len: cfg.max_len,
		input_names,
		output_names,
	}))
}

/// Kind-specific metadata, read from the config files and a built session.
fn build_meta(cfg: &ModelConfig, resolved: &hub::Resolved, encoder: &Encoder, session: &Session) -> Result<Meta> {
	Ok(match cfg.kind {
		Kind::Embedding => Meta::Embedding {
			pooling: resolve_pooling(cfg, resolved),
			output: pick_output(session, &[graph_pooling::POOLED_OUTPUT, "sentence_embedding", "_pooler_output", "pooler_output", "last_hidden_state", "token_embeddings", "output0"])
				.ok_or_else(|| Error::Config(format!("model '{}': no outputs", cfg.name)))?,
			normalize: cfg.normalize,
			dimensions: cfg.dimensions,
		},
		Kind::Rerank => {
			let (yes_id, no_id) = match cfg.scoring {
				Scoring::YesNo => (
					Some(encoder.single_token_id("yes").ok_or_else(|| Error::Config(format!("model '{}': tokenizer cannot encode 'yes'", cfg.name)))?),
					Some(encoder.single_token_id("no").ok_or_else(|| Error::Config(format!("model '{}': tokenizer cannot encode 'no'", cfg.name)))?),
				),
				Scoring::Auto => (encoder.single_token_id("yes"), encoder.single_token_id("no")),
				_ => (None, None),
			};
			Meta::Rerank {
				scoring: cfg.scoring,
				yes_id,
				no_id,
				output: pick_output(session, &["logits", "output0", "output", "scores"])
					.ok_or_else(|| Error::Config(format!("model '{}': no outputs", cfg.name)))?,
			}
		}
		Kind::Zeroshot => {
			let id2label = id2label_from_config(resolved)
				.ok_or_else(|| Error::Config(format!("model '{}': config.json must contain id2label", cfg.name)))?;
			let entailment = find_label(&id2label, &cfg.entailment_label)
				.ok_or_else(|| Error::Config(format!("model '{}': no entailment label found in {:?}", cfg.name, id2label)))?;
			let contradiction = find_label(&id2label, &cfg.contradiction_label)
				.ok_or_else(|| Error::Config(format!("model '{}': no contradiction label found in {:?}", cfg.name, id2label)))?;
			Meta::Zeroshot {
				entailment,
				contradiction,
				template: cfg.hypothesis_template.clone(),
				output: pick_output(session, &["logits", "output0", "output"])
					.ok_or_else(|| Error::Config(format!("model '{}': no outputs", cfg.name)))?,
			}
		}
		Kind::Pii => {
			let id2label = id2label_from_config(resolved)
				.ok_or_else(|| Error::Config(format!("model '{}': config.json must contain id2label for PII decoding", cfg.name)))?;
			Meta::Pii {
				id2label,
				output: pick_output(session, &["logits", "output0", "output"])
					.ok_or_else(|| Error::Config(format!("model '{}': no outputs", cfg.name)))?,
			}
		}
	})
}

/// GPU execution providers, which `dtype: auto` runs in fp16.
const GPU_EPS: &[&str] = &["cuda", "tensorrt", "nvrtx"];

/// The config the published graph is picked with: `dtype: auto` led by a GPU
/// provider prefers the fp16 graph (fp32 if the repo has none) for half the
/// memory traffic and tensor-core math.
fn graph_preference<'a>(cfg: &'a ModelConfig, eps: &[String]) -> std::borrow::Cow<'a, ModelConfig> {
	if cfg.dtype == Dtype::Auto && eps.first().is_some_and(|e| GPU_EPS.contains(&e.as_str())) {
		let mut fp16 = cfg.clone();
		fp16.dtype = Dtype::Fp16;
		return std::borrow::Cow::Owned(fp16);
	}
	std::borrow::Cow::Borrowed(cfg)
}

fn resolve_pooling(cfg: &ModelConfig, resolved: &hub::Resolved) -> Pooling {
	match cfg.pooling {
		Pooling::Auto => pooling_from_st_config(resolved).unwrap_or(Pooling::Mean),
		p => p,
	}
}

/// Outputs an export can already provide pooled; `build_meta` prefers them.
const POOLED_EXPORT_OUTPUTS: &[&str] = &["sentence_embedding", "_pooler_output", "pooler_output"];

/// For embedding graphs that only output token states, the same graph with the
/// pooling appended (see `graph_pooling`): by default on accelerator execution
/// providers, where it saves copying [B,T,D] back to the host; `pooling_in_graph`
/// forces it on or off. Falls back to the graph as is (pooling on the CPU).
fn pool_in_graph(cfg: &ModelConfig, resolved: &hub::Resolved, graph: PathBuf, eps: &[String], hf_cache: Option<&Path>) -> (PathBuf, bool) {
	if cfg.kind != Kind::Embedding || !cfg.pooling_in_graph.unwrap_or_else(|| eps.iter().any(|e| e != "cpu")) {
		return (graph, false);
	}
	let outputs = match graph_pooling::output_names(&graph) {
		Ok(o) => o,
		Err(e) => {
			tracing::warn!(model = cfg.name.as_str(), error = %e, "cannot read graph outputs; pooling on the CPU");
			return (graph, false);
		}
	};
	if outputs.iter().any(|o| POOLED_EXPORT_OUTPUTS.contains(&o.as_str())) {
		return (graph, false);
	}
	let Some(token_output) = outputs.iter().find(|o| *o == "last_hidden_state" || *o == "token_embeddings") else {
		return (graph, false);
	};
	match graph_pooling::cached_pooled(&graph, token_output, resolve_pooling(cfg, resolved), &hub::cache_root(hf_cache)) {
		Ok(pooled) => (pooled, true),
		Err(e) => {
			tracing::warn!(model = cfg.name.as_str(), error = %e, "pooling in the graph failed; pooling on the CPU");
			(graph, false)
		}
	}
}

fn build_extract(meta: &Meta) -> Arc<Extract> {
	match meta {
		Meta::Embedding { pooling, output, normalize, dimensions } => pipeline::embedding::extractor(*pooling, output.clone(), *normalize, *dimensions),
		Meta::Rerank { scoring, yes_id, no_id, output } => pipeline::rerank::extractor(*scoring, *yes_id, *no_id, output.clone()),
		Meta::Zeroshot { output, .. } => pipeline::zeroshot::extractor(output.clone()),
		Meta::Pii { id2label, output } => pipeline::pii::extractor(id2label.len(), output.clone()),
	}
}

/// The graph the sessions run: the resolved file, or the server's own int8
/// rewrite of it (`dtype: int8`, or `auto` with CPU-only execution) once it
/// passed the quality gate. Any quantization problem falls back to the
/// resolved file. Returns the path and a label for the startup log.
fn choose_graph(cfg: &ModelConfig, resolved: &hub::Resolved, encoder: &Encoder, eps: &[String], hf_cache: Option<&Path>) -> (PathBuf, &'static str) {
	let wanted = match cfg.dtype {
		Dtype::Int8 => true,
		Dtype::Auto => eps.iter().all(|e| e == "cpu"),
		Dtype::Fp32 | Dtype::Fp16 => false,
	};
	let published = (resolved.model.clone(), "as published");
	if !wanted {
		return published;
	}
	let int8 = match quantize::cached_int8(&resolved.model, &hub::cache_root(hf_cache)) {
		Ok(Some(path)) => path,
		Ok(None) => return (resolved.model.clone(), "as published (nothing to quantize)"),
		Err(e) => {
			tracing::warn!(model = cfg.name.as_str(), error = %e, "int8 quantization failed; serving the published graph");
			return published;
		}
	};
	match quality_gate(cfg, resolved, encoder, &int8) {
		Ok(true) => (int8, "int8 (per-channel, quantized by rs-infer)"),
		Ok(false) => published,
		Err(e) => {
			tracing::warn!(model = cfg.name.as_str(), error = %e, "int8 quality check failed to run; serving the published graph");
			published
		}
	}
}

/// Compares the published graph and its int8 rewrite on built-in calibration
/// inputs, once per (graph, kind): the verdict is cached next to the int8 graph.
fn quality_gate(cfg: &ModelConfig, resolved: &hub::Resolved, encoder: &Encoder, int8: &Path) -> Result<bool> {
	let verdict_file = int8.with_file_name(format!("quality-{}.txt", cfg.kind.as_str()));
	if let Ok(verdict) = std::fs::read_to_string(&verdict_file) {
		tracing::info!(model = cfg.name.as_str(), verdict = verdict.trim(), "int8 quality check (cached)");
		return Ok(verdict.starts_with("pass"));
	}
	let enc = quantize::calibration(encoder, cfg.kind)?;
	let mut reference_session = new_session(&resolved.model, cfg)?;
	let extract = build_extract(&build_meta(cfg, resolved, encoder, &reference_session)?);
	let inputs = InputSpec::of(&reference_session)?.build(&enc)?;
	let reference = extract(&mut reference_session, inputs, &enc)?;
	drop(reference_session);
	let mut candidate_session = new_session(int8, cfg)?;
	let inputs = InputSpec::of(&candidate_session)?.build(&enc)?;
	let candidate = extract(&mut candidate_session, inputs, &enc)?;
	let check = quantize::compare(&reference, &candidate, &enc.attention_mask)?;
	let verdict = format!("{} {} {:.4} (threshold {})", if check.pass { "pass" } else { "fail" }, check.metric, check.worst, check.threshold);
	if let Err(e) = std::fs::write(&verdict_file, &verdict) {
		tracing::warn!(error = %e, "cannot cache the int8 quality verdict");
	}
	if check.pass {
		tracing::info!(model = cfg.name.as_str(), verdict, "int8 quality check passed");
	} else {
		tracing::warn!(model = cfg.name.as_str(), verdict, "int8 output drifts too far from the published graph; serving it instead (set dtype: fp32 to skip this check)");
	}
	Ok(check.pass)
}

#[cfg(test)]
#[path = "tests/registry_tests.rs"]
mod tests;
