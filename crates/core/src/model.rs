use std::{collections::HashMap, sync::Arc};

use ort::session::Session;

use crate::{
	batcher::Batcher,
	config::{Kind, ModelConfig, Pooling, Scoring},
	hub::Resolved,
	pipeline::Extract,
	pool::SessionPool,
	tokenize::{Encoder, InputSpec},
};

/// Which session output tensor a pipeline consumes.
#[derive(Debug, Clone)]
pub struct OutSel(pub String);

#[derive(Debug)]
pub enum Meta {
	Embedding {
		pooling: Pooling,
		output: OutSel,
		normalize: bool,
		dimensions: Option<usize>,
	},
	Rerank {
		scoring: Scoring,
		yes_id: Option<u32>,
		no_id: Option<u32>,
		output: OutSel,
	},
	Zeroshot {
		entailment: usize,
		contradiction: usize,
		template: String,
		output: OutSel,
	},
	Pii {
		/// id -> label, e.g. "B-person".
		id2label: Vec<String>,
		output: OutSel,
	},
}

pub struct LoadedModel {
	pub cfg: ModelConfig,
	pub encoder: Encoder,
	pub pool: Arc<SessionPool>,
	/// Forward pass + post-processing for a padded batch of this model's rows.
	pub extract: Arc<Extract>,
	/// How to turn a padded batch into this model's input tensors.
	pub inputs: Arc<InputSpec>,
	/// Cross-request batcher, unless `batching.enabled` is false.
	pub batcher: Option<Arc<Batcher>>,
	pub meta: Meta,
	pub source: String,
	pub eps: Vec<String>,
	pub max_len: Option<usize>,
	pub input_names: Vec<String>,
	pub output_names: Vec<String>,
}

impl LoadedModel {
	pub fn kind(&self) -> Kind {
		self.cfg.kind
	}

	pub fn name(&self) -> &str {
		&self.cfg.name
	}
}

/// Picks the output to read from a session, preferring conventional names.
pub fn pick_output(session: &Session, preferred: &[&str]) -> Option<OutSel> {
	let names: Vec<String> = session.outputs().iter().map(|o| o.name().to_string()).collect();
	for p in preferred {
		if names.iter().any(|n| n == p) {
			return names.iter().find(|n| n == p).cloned().map(OutSel);
		}
	}
	names.first().cloned().map(OutSel)
}

pub fn id2label_from_config(resolved: &Resolved) -> Option<Vec<String>> {
	let path = resolved.config_json.as_ref()?;
	let raw = std::fs::read_to_string(path).ok()?;
	let value: serde_json::Value = serde_json::from_str(&raw).ok()?;
	let map = value.get("id2label")?.as_object()?;
	let mut pairs: Vec<(usize, String)> = Vec::with_capacity(map.len());
	for (k, v) in map {
		let Ok(id) = k.parse::<usize>() else { continue };
		let Some(s) = v.as_str() else { continue };
		pairs.push((id, s.to_string()));
	}
	if pairs.is_empty() {
		return None;
	}
	pairs.sort_unstable();
	let n = pairs.last()?.0 + 1;
	let mut out = vec![String::new(); n];
	for (id, label) in pairs {
		out[id] = label;
	}
	Some(out)
}

pub fn pooling_from_st_config(resolved: &Resolved) -> Option<Pooling> {
	let path = resolved.pooling_config.as_ref()?;
	let raw = std::fs::read_to_string(path).ok()?;
	let value: serde_json::Value = serde_json::from_str(&raw).ok()?;
	let b = |k: &str| value.get(k).and_then(|v| v.as_bool()).unwrap_or(false);
	if b("pooling_mode_cls_token") {
		Some(Pooling::Cls)
	} else if b("pooling_mode_lasttoken") {
		Some(Pooling::Last)
	} else if b("pooling_mode_mean_tokens") || b("pooling_mode_mean_sqrt_len_tokens") || b("pooling_mode_max_tokens") {
		Some(Pooling::Mean)
	} else {
		None
	}
}

pub fn find_label(labels: &[String], wanted: &str) -> Option<usize> {
	let w = wanted.to_ascii_lowercase();
	labels.iter().position(|l| l.to_ascii_lowercase() == w).or_else(|| labels.iter().position(|l| l.to_ascii_lowercase().contains(&w)))
}

pub type ConfigJson = HashMap<String, serde_json::Value>;
