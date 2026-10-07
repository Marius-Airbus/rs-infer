use std::{collections::BTreeSet, path::PathBuf};

use serde::Deserialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
	Embedding,
	Rerank,
	Zeroshot,
	Pii,
}

impl Kind {
	pub fn as_str(self) -> &'static str {
		match self {
			Kind::Embedding => "embedding",
			Kind::Rerank => "rerank",
			Kind::Zeroshot => "zeroshot",
			Kind::Pii => "pii",
		}
	}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EpName {
	Cpu,
	Coreml,
	Cuda,
	Tensorrt,
	Nvrtx,
}

impl EpName {
	pub fn as_str(self) -> &'static str {
		match self {
			EpName::Cpu => "cpu",
			EpName::Coreml => "coreml",
			EpName::Cuda => "cuda",
			EpName::Tensorrt => "tensorrt",
			EpName::Nvrtx => "nvrtx",
		}
	}
}

impl std::fmt::Display for EpName {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.write_str(self.as_str())
	}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Pooling {
	#[default]
	Auto,
	Cls,
	Mean,
	Last,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Scoring {
	/// Auto-detect from output shape; all auto modes produce scores in (0,1):
	/// [B,1] -> sigmoid(logit), [B,2] -> softmax[1], [B,S,V] -> yes_no.
	#[default]
	Auto,
	/// Raw logit from a single-output cross-encoder (unbounded; opt-in only).
	Logit,
	Sigmoid,
	/// Two-output cross-encoder (negative, positive): P(positive).
	Softmax,
	/// Qwen3-Reranker style: P("yes") over vocab logits at the last token.
	YesNo,
}

/// ONNX Runtime session log verbosity (ORT's native level filter).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OrtLogLevel {
	Verbose,
	Info,
	#[default]
	Warn,
	Error,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoreMlComputeUnits {
	#[default]
	All,
	CpuAndGpu,
	CpuAndNe,
	CpuOnly,
}

/// Precision of the graph the model runs.
/// - `auto`: on CPU-only execution, the server's own per-channel int8 rewrite of
///   the fp32 graph (cached, kept only if it passes a quality check against
///   fp32, see `quantize`); with GPU/CoreML execution providers, the published
///   graph (fp32 first, else fp16 > quantized files).
/// - `int8`: that int8 rewrite whatever the providers (the publisher's int8
///   files if the repo has no fp32 graph).
/// - `fp32` / `fp16`: the published graph of that format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Dtype {
	#[default]
	Auto,
	Fp32,
	Fp16,
	Int8,
}

/// Cross-request batching (on by default, every model kind): rows from concurrent
/// requests are sorted by length and packed into forwards of up to `max_rows`
/// rows and `max_tokens` padded tokens, spread over the free session replicas. A
/// row arriving to an empty queue is dispatched immediately, so batching adds no
/// latency; it amortizes per-forward cost when requests overlap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Batching {
	/// `false`: every request runs as its own forward(s) (`max_batch`-row slices).
	pub enabled: bool,
	/// Max rows per assembled batch.
	pub max_rows: usize,
	/// Max padded tokens (rows x longest row) per batch; bounds activation memory.
	pub max_tokens: usize,
	/// Max rows queued (in + waiting) before shedding with 429; a request with
	/// more rows than this bypasses the queue.
	pub queue_rows: usize,
}

impl Default for Batching {
	fn default() -> Self {
		Self { enabled: true, max_rows: 64, max_tokens: 8192, queue_rows: 1024 }
	}
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
	#[serde(default)]
	pub server: ServerConfig,
	pub models: Vec<ModelConfig>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ServerConfig {
	pub bind: String,
	/// Hard cap on request body size.
	pub max_body_mb: usize,
	/// Overall per-request timeout.
	pub request_timeout_ms: u64,
	/// Max HTTP connections waiting for a session slot, per model, before 429.
	pub max_queue: usize,
	pub queue_timeout_ms: u64,
	/// Directory used by hf-hub for downloads (also honors HF_HOME).
	pub hf_cache_dir: Option<PathBuf>,
	/// Threads of the ONNX Runtime pool shared by every session of every model;
	/// 0 = number of physical cores. A model with `intra_threads > 0` gets its
	/// own pool instead.
	pub threads: usize,
}

impl Default for ServerConfig {
	fn default() -> Self {
		Self {
			bind: "0.0.0.0:8080".into(),
			max_body_mb: 32,
			request_timeout_ms: 120_000,
			max_queue: 256,
			queue_timeout_ms: 30_000,
			hf_cache_dir: None,
			threads: 0,
		}
	}
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
	pub name: String,
	pub kind: Kind,
	/// Local directory containing model.onnx + tokenizer.json.
	#[serde(default)]
	pub path: Option<PathBuf>,
	/// Hugging Face repo id; downloaded into the cache at startup.
	#[serde(default)]
	pub hf: Option<String>,
	#[serde(default = "default_revision")]
	pub revision: String,
	/// Subdirectory inside the repo used as fallback for model files, e.g. "onnx".
	#[serde(default)]
	pub subfolder: Option<String>,
	/// Pin an exact model filename (e.g. "onnx/model_quantized.onnx");
	/// overrides the default model.onnx > fp16 > quantized preference.
	#[serde(default)]
	pub file: Option<String>,
	/// Preferred graph variant (weight format); reorders the file candidates.
	#[serde(default)]
	pub dtype: Dtype,
	/// HF repo to take `tokenizer.json` (and optionally `config.json`) from,
	/// for model-only ONNX repos. Defaults to `hf` itself.
	#[serde(default)]
	pub tokenizer_hf: Option<String>,
	#[serde(default)]
	pub max_len: Option<usize>,
	/// Number of concurrent sessions (pool replicas) for this model.
	#[serde(default = "default_replicas")]
	pub replicas: usize,
	/// 0 = sessions run on the shared thread pool (`server.threads`); > 0 = each
	/// session of this model gets its own pool of that many threads.
	#[serde(default)]
	pub intra_threads: usize,
	/// Execution provider priority list; entries not compiled in are skipped with a warning.
	#[serde(default)]
	pub eps: Vec<EpName>,
	/// Make this model the default for its kind even if loaded later.
	#[serde(default)]
	pub default: bool,
	/// Cross-request dynamic batching (on by default; `batching: { enabled: false }` to opt out).
	#[serde(default)]
	pub batching: Batching,
	/// Max rows per forward pass; a bigger request runs as successive slices, so
	/// one request cannot blow up activation memory.
	#[serde(default = "default_max_batch")]
	pub max_batch: usize,

	// --- embedding ---
	#[serde(default)]
	pub pooling: Pooling,
	#[serde(default = "default_true")]
	pub normalize: bool,
	/// Matryoshka truncation; requires `normalize` to be applied after truncation.
	#[serde(default)]
	pub dimensions: Option<usize>,

	// --- rerank ---
	#[serde(default)]
	pub scoring: Scoring,

	// --- zeroshot ---
	#[serde(default = "default_hypothesis")]
	pub hypothesis_template: String,
	#[serde(default = "default_entail_label")]
	pub entailment_label: String,
	#[serde(default = "default_contradiction_label")]
	pub contradiction_label: String,

	// --- pii ---
	/// Minimum token probability; below this the token is treated as outside any entity.
	#[serde(default = "default_threshold")]
	pub threshold: f64,

	// --- EP options ---
	/// ORT session log level: verbose | info | warn | error. "error" silences
	/// CoreML/TensorRT graph-partition chatter (e.g. "Some nodes were not
	/// assigned to the preferred execution providers").
	#[serde(default)]
	pub ort_log_level: OrtLogLevel,
	#[serde(default)]
	pub coreml_compute_units: CoreMlComputeUnits,
	#[serde(default)]
	pub trt_engine_cache: Option<PathBuf>,
	#[serde(default)]
	pub device_id: i32,

	/// ORT profiling file prefix, set programmatically by the `profile` command;
	/// not configurable from YAML.
	#[serde(skip, default)]
	pub profiling_prefix: Option<PathBuf>,
}

fn default_revision() -> String {
	"main".into()
}
fn default_replicas() -> usize {
	2
}
fn default_max_batch() -> usize {
	32
}
fn default_true() -> bool {
	true
}
fn default_hypothesis() -> String {
	"The text is about {label}.".into()
}
fn default_entail_label() -> String {
	"entailment".into()
}
fn default_contradiction_label() -> String {
	"contradiction".into()
}
fn default_threshold() -> f64 {
	0.5
}

impl Default for ModelConfig {
	fn default() -> Self {
		Self {
			name: String::new(),
			kind: Kind::Embedding,
			path: None,
			hf: None,
			revision: default_revision(),
			subfolder: None,
			file: None,
			dtype: Dtype::Auto,
			tokenizer_hf: None,
			max_len: None,
			replicas: default_replicas(),
			intra_threads: 0,
			eps: vec![],
			default: false,
			batching: Batching::default(),
			max_batch: default_max_batch(),
			pooling: Pooling::Auto,
			normalize: true,
			dimensions: None,
			scoring: Scoring::Auto,
			hypothesis_template: default_hypothesis(),
			entailment_label: default_entail_label(),
			contradiction_label: default_contradiction_label(),
			threshold: default_threshold(),
			ort_log_level: OrtLogLevel::Warn,
			coreml_compute_units: CoreMlComputeUnits::All,
			trt_engine_cache: None,
			device_id: 0,
			profiling_prefix: None,
		}
	}
}

#[cfg(test)]
impl ModelConfig {
	#[doc(hidden)]
	pub fn default_for_test(file: Option<&str>) -> Self {
		Self { name: "test".into(), file: file.map(str::to_string), replicas: 1, ..Default::default() }
	}
}

impl ModelConfig {
	pub fn validate(&self) -> crate::Result<()> {
		if self.path.is_none() && self.hf.is_none() {
			return Err(crate::Error::Config(format!(
				"model '{}': one of `path` or `hf` must be set",
				self.name
			)));
		}
		let mut seen = BTreeSet::new();
		for ep in &self.eps {
			if !seen.insert(*ep) {
				return Err(crate::Error::Config(format!(
					"model '{}': duplicate ep '{}'",
					self.name, ep
				)));
			}
		}
		if self.replicas == 0 {
			return Err(crate::Error::Config(format!("model '{}': replicas must be >= 1", self.name)));
		}
		if self.max_batch == 0 {
			return Err(crate::Error::Config(format!("model '{}': max_batch must be >= 1", self.name)));
		}
		Ok(())
	}
}

#[cfg(test)]
#[path = "tests/config_tests.rs"]
mod tests;
