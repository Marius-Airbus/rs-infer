use std::{collections::HashMap, sync::Arc};

use crate::{
	config::{Config, Kind, ModelConfig, Pooling, Scoring},
	ep::{new_session, resolve_eps},
	hub,
	model::{find_label, id2label_from_config, pick_output, pooling_from_st_config, Meta},
	pool::SessionPool,
	tokenize::Encoder,
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

	/// Spawns dynamic-batching workers for embedding models that have `batching:`
	/// configured. Must run on the tokio runtime (models load on blocking threads).
	pub fn start_batchers(&self) {
		let mut n = 0;
		for m in self.models.values() {
			if let Some(b) = &m.batcher {
				b.start();
				n += 1;
			}
		}
		if n > 0 {
			tracing::info!(models = n, "dynamic embedding batching enabled");
		}
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

	let resolved = hub::resolve(cfg, hf_cache)?;
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

	let (eps, _) = resolve_eps(cfg);
	let t_sessions = std::time::Instant::now();
	let mut sessions = Vec::with_capacity(cfg.replicas);
	for _ in 0..cfg.replicas {
		sessions.push(new_session(&resolved.model, cfg)?);
	}
	let session_ms = t_sessions.elapsed().as_millis();
	let first = sessions.first().expect("replicas >= 1, validated");
	let input_names = first.inputs().iter().map(|i| i.name().to_string()).collect();
	let output_names = first.outputs().iter().map(|o| o.name().to_string()).collect();

	let meta = match cfg.kind {
		Kind::Embedding => Meta::Embedding {
			pooling: match cfg.pooling {
				Pooling::Auto => pooling_from_st_config(&resolved).unwrap_or(Pooling::Mean),
				p => p,
			},
			output: pick_output(first, &["sentence_embedding", "_pooler_output", "pooler_output", "last_hidden_state", "token_embeddings", "output0"])
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
				output: pick_output(first, &["logits", "output0", "output", "scores"])
					.ok_or_else(|| Error::Config(format!("model '{}': no outputs", cfg.name)))?,
			}
		}
		Kind::Zeroshot => {
			let id2label = id2label_from_config(&resolved)
				.ok_or_else(|| Error::Config(format!("model '{}': config.json must contain id2label", cfg.name)))?;
			let entailment = find_label(&id2label, &cfg.entailment_label)
				.ok_or_else(|| Error::Config(format!("model '{}': no entailment label found in {:?}", cfg.name, id2label)))?;
			let contradiction = find_label(&id2label, &cfg.contradiction_label)
				.ok_or_else(|| Error::Config(format!("model '{}': no contradiction label found in {:?}", cfg.name, id2label)))?;
			Meta::Zeroshot {
				entailment,
				contradiction,
				template: cfg.hypothesis_template.clone(),
				output: pick_output(first, &["logits", "output0", "output"])
					.ok_or_else(|| Error::Config(format!("model '{}': no outputs", cfg.name)))?,
			}
		}
		Kind::Pii => {
			let id2label = id2label_from_config(&resolved)
				.ok_or_else(|| Error::Config(format!("model '{}': config.json must contain id2label for PII decoding", cfg.name)))?;
			Meta::Pii {
				id2label,
				output: pick_output(first, &["logits", "output0", "output"])
					.ok_or_else(|| Error::Config(format!("model '{}': no outputs", cfg.name)))?,
			}
		}
	};

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
		session_build_ms = session_ms,
		elapsed_ms = t_start.elapsed().as_millis(),
		ram_used_mb = rss_used,
		ram_total_mb = rss_after,
		"model loaded"
	);
	let pool = Arc::new(SessionPool::new(sessions, max_queue));
	// Cross-request batching is opt-in per model (`batching:`) and only for
	// embeddings: rows are per-text and order-independent, unlike pair/token
	// classification outputs.
	let batcher = match (&meta, cfg.batching.as_ref()) {
		(Meta::Embedding { pooling, output, normalize, dimensions }, Some(settings)) => Some(crate::batcher::EmbedBatcher::new(
			pool.clone(),
			crate::batcher::EmbedMeta { pooling: *pooling, output: output.clone(), normalize: *normalize, dimensions: *dimensions },
			*settings,
		)),
		_ => None,
	};
	Ok(Arc::new(LoadedModel {
		cfg: cfg.clone(),
		encoder,
		pool,
		batcher,
		meta,
		source: resolved.source,
		eps,
		max_len: cfg.max_len,
		input_names,
		output_names,
	}))
}
