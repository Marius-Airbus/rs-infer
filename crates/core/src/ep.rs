#[allow(unused_imports)]
use ort::{
	ep::{self, ExecutionProvider, ExecutionProviderDispatch},
	logging::LogLevel,
	session::{
		builder::{GraphOptimizationLevel, SessionBuilder},
		Session,
	},
};

use crate::{
	config::{EpName, ModelConfig},
	Error, Result,
};

static SHARED_POOL_THREADS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();

/// Installs the ONNX Runtime intra-op thread pool shared by all sessions
/// (`threads` = 0: physical cores) and returns its size. Must run before any
/// session is built; later calls return the existing size.
///
/// One shared pool sized to physical cores beats per-session pools: replicas'
/// pools oversubscribe the CPU, and threads on SMT siblings / efficiency cores
/// make every parallel op wait for the slowest one. On a 14-core i7-13850HX
/// (e5-small int8, 1-text requests, 4 replicas) it gave 1020 req/s vs 655 with
/// four 4-thread pools and 60 with the former 2 x all-logical-cores default.
pub fn init_shared_thread_pool(threads: usize) -> Result<usize> {
	if let Some(&n) = SHARED_POOL_THREADS.get() {
		return Ok(n);
	}
	let n = if threads > 0 { threads } else { num_cpus::get_physical().max(1) };
	let options = ort::environment::GlobalThreadPoolOptions::default().with_intra_threads(n)?;
	if !ort::init().with_global_thread_pool(options).commit() {
		return Err(Error::Config("ONNX Runtime was initialized before the shared thread pool could be installed".into()));
	}
	let _ = SHARED_POOL_THREADS.set(n);
	Ok(n)
}

/// All EPs known to the runtime, with compile-time availability.
pub const ALL_EPS: &[EpName] = &[EpName::Cpu, EpName::Coreml, EpName::Cuda, EpName::Tensorrt, EpName::Nvrtx, EpName::Openvino];

pub fn compiled_in(ep: EpName) -> bool {
	match ep {
		EpName::Cpu => true,
		#[cfg(feature = "ep-coreml")]
		EpName::Coreml => true,
		#[cfg(feature = "ep-cuda")]
		EpName::Cuda => true,
		#[cfg(feature = "ep-tensorrt")]
		EpName::Tensorrt => true,
		#[cfg(feature = "ep-nvrtx")]
		EpName::Nvrtx => true,
		#[cfg(feature = "ep-openvino")]
		EpName::Openvino => true,
		_ => false,
	}
}

/// Whether the linked ONNX Runtime build actually contains this EP.
pub fn runtime_available(ep: EpName) -> bool {
	match ep {
		EpName::Cpu => true,
		#[cfg(feature = "ep-coreml")]
		EpName::Coreml => ep::CoreML::default().is_available().unwrap_or(false),
		#[cfg(feature = "ep-cuda")]
		EpName::Cuda => ep::CUDA::default().is_available().unwrap_or(false),
		#[cfg(feature = "ep-tensorrt")]
		EpName::Tensorrt => ep::TensorRT::default().is_available().unwrap_or(false),
		#[cfg(feature = "ep-nvrtx")]
		EpName::Nvrtx => ep::NVRTX::default().is_available().unwrap_or(false),
		#[cfg(feature = "ep-openvino")]
		EpName::Openvino => ep::OpenVINO::default().is_available().unwrap_or(false),
		_ => false,
	}
}

/// TensorRT optimization profile: min/opt/max shapes of the token inputs.
#[cfg_attr(not(feature = "ep-tensorrt"), allow(dead_code))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TrtProfile {
	pub min: String,
	pub opt: String,
	pub max: String,
}

/// Explicit TensorRT profile for graphs whose inputs are all `[batch, tokens]`
/// token inputs, sized from the batch limits (`batching.max_rows` /
/// `max_batch`) and `max_len` (512 if unset), so TensorRT builds one engine
/// covering every batch the server sends instead of rebuilding when a new
/// shape exceeds the previous range. `None` (TensorRT's own dynamic handling)
/// for other graphs, e.g. decoder exports with KV-cache inputs.
pub(crate) fn trt_profile(model_path: &std::path::Path, cfg: &ModelConfig) -> Option<TrtProfile> {
	const TOKEN_INPUTS: &[&str] = &["input_ids", "attention_mask", "token_type_ids", "position_ids"];
	let inputs = crate::graph_pooling::input_names(model_path).ok()?;
	if inputs.is_empty() || !inputs.iter().all(|i| TOKEN_INPUTS.contains(&i.as_str())) {
		return None;
	}
	let max_rows = cfg.batching.max_rows.max(cfg.max_batch).max(1);
	let max_tokens = cfg.max_len.unwrap_or(512).max(1);
	let shapes = |rows: usize, tokens: usize| inputs.iter().map(|i| format!("{i}:{rows}x{tokens}")).collect::<Vec<_>>().join(",");
	Some(TrtProfile {
		min: shapes(1, 1),
		opt: shapes((max_rows / 2).max(1), (max_tokens / 2).max(1)),
		max: shapes(max_rows, max_tokens),
	})
}

#[allow(unused_variables)]
fn dispatch(cfg: &ModelConfig, name: EpName, trt: Option<&TrtProfile>) -> Option<ExecutionProviderDispatch> {
	match name {
		EpName::Cpu => Some(ep::CPU::default().build()),
		#[cfg(feature = "ep-coreml")]
		EpName::Coreml => {
			use crate::config::CoreMlComputeUnits;
			let units = match cfg.coreml_compute_units {
				CoreMlComputeUnits::All => ep::coreml::ComputeUnits::All,
				CoreMlComputeUnits::CpuAndGpu => ep::coreml::ComputeUnits::CPUAndGPU,
				CoreMlComputeUnits::CpuAndNe => ep::coreml::ComputeUnits::CPUAndNeuralEngine,
				CoreMlComputeUnits::CpuOnly => ep::coreml::ComputeUnits::CPUOnly,
			};
			Some(ep::CoreML::default().with_compute_units(units).with_model_cache_dir(std::env::temp_dir().join("rsinfer-coreml").display().to_string()).build())
		}
		#[cfg(feature = "ep-cuda")]
		// Grow the GPU arena by what is requested rather than doubling it: batches
		// change shape all the time, and doubling strands VRAM.
		EpName::Cuda => Some(
			ep::CUDA::default()
				.with_device_id(cfg.device_id)
				.with_arena_extend_strategy(ep::ArenaExtendStrategy::SameAsRequested)
				.build(),
		),
		#[cfg(feature = "ep-tensorrt")]
		EpName::Tensorrt => {
			// fp16 engines (tensor cores) with LayerNorm kept in fp32, where fp16
			// overflows in transformer encoders; engines and kernel timings cached.
			let mut b = ep::TensorRT::default()
				.with_device_id(cfg.device_id)
				.with_fp16(true)
				.with_layer_norm_fp32_fallback(true)
				.with_engine_cache(true)
				.with_timing_cache(true);
			if let Some(dir) = &cfg.trt_engine_cache {
				b = b.with_engine_cache_path(dir.display().to_string()).with_timing_cache_path(dir.display().to_string());
			}
			if let Some(p) = trt {
				b = b.with_profile_min_shapes(&p.min).with_profile_opt_shapes(&p.opt).with_profile_max_shapes(&p.max);
			}
			Some(b.build())
		}
		#[cfg(feature = "ep-nvrtx")]
		EpName::Nvrtx => {
			let mut b = ep::NVRTX::default().with_device_id(cfg.device_id as u32);
			if let Some(dir) = &cfg.trt_engine_cache {
				b = b.with_runtime_cache_path(dir.display().to_string());
			}
			Some(b.build())
		}
		// fp32 graphs only: OpenVINO runs dynamic int8 (MatMulInteger) 2-4x slower than
		// ORT's CPU kernels, which is why `dtype: auto` keeps the published graph here.
		#[cfg(feature = "ep-openvino")]
		EpName::Openvino => Some(
			ep::OpenVINO::default()
				.with_device_type(&cfg.openvino_device)
				.with_cache_dir(std::env::temp_dir().join("rsinfer-openvino").display().to_string())
				.build(),
		),
		_ => None,
	}
}

/// Resolves the effective EP list (request order, compiled-in + runtime-available, CPU appended)
/// together with the registrations that can actually be attempted.
pub fn resolve_eps(cfg: &ModelConfig) -> (Vec<String>, Vec<ExecutionProviderDispatch>) {
	resolve_eps_with(cfg, None)
}

fn resolve_eps_with(cfg: &ModelConfig, trt: Option<&TrtProfile>) -> (Vec<String>, Vec<ExecutionProviderDispatch>) {
	let requested: Vec<EpName> = if cfg.eps.is_empty() { vec![EpName::Cpu] } else { cfg.eps.clone() };
	let mut used = Vec::new();
	let mut dispatches = Vec::new();
	for name in requested {
		if !compiled_in(name) {
			tracing::warn!(model = cfg.name.as_str(), ep = %name, "execution provider not compiled in; enable the matching cargo feature to use it");
			continue;
		}
		if !runtime_available(name) {
			tracing::warn!(model = cfg.name.as_str(), ep = %name, "ONNX Runtime build lacks this execution provider; skipping");
			continue;
		}
		match dispatch(cfg, name, trt) {
			Some(d) => {
				used.push(name.as_str().to_owned());
				dispatches.push(d);
			}
			None => tracing::warn!(model = cfg.name.as_str(), ep = %name, "execution provider unavailable on this platform; skipping"),
		}
	}
	if !dispatches.is_empty() && !used.iter().any(|u| u == "cpu") {
		used.push("cpu".into());
	}
	(used, dispatches)
}

/// Creates one session with the resolved execution providers + graph optimizations.
pub fn new_session(model_path: &std::path::Path, cfg: &ModelConfig) -> Result<Session> {
	let trt = (cfg.eps.contains(&EpName::Tensorrt) && compiled_in(EpName::Tensorrt)).then(|| trt_profile(model_path, cfg)).flatten();
	let (used, dispatches) = resolve_eps_with(cfg, trt.as_ref());
	tracing::debug!(model = cfg.name.as_str(), eps = ?used, trt_profile = ?trt, "building session");

	let mut builder: SessionBuilder = Session::builder()?
		.with_optimization_level(GraphOptimizationLevel::Level3)
		.map_err(|e| Error::Ort(e.into()))?;
	let log_level = match cfg.ort_log_level {
		crate::config::OrtLogLevel::Verbose => LogLevel::Verbose,
		crate::config::OrtLogLevel::Info => LogLevel::Info,
		crate::config::OrtLogLevel::Warn => LogLevel::Warning,
		crate::config::OrtLogLevel::Error => LogLevel::Error,
	};
	builder = builder.with_log_level(log_level).map_err(|e| Error::Ort(e.into()))?;
	if let Some(prefix) = &cfg.profiling_prefix {
		builder = builder.with_profiling(prefix).map_err(|e| Error::Ort(e.into()))?;
	}
	// intra_threads = 0: run on the shared pool (see init_shared_thread_pool), or
	// ORT's per-session default (physical cores) if none was installed.
	// intra_threads > 0: this session gets its own pool of exactly that size.
	if cfg.intra_threads > 0 {
		builder = builder.with_independent_thread_pool().map_err(|e| Error::Ort(e.into()))?;
		builder = builder.with_intra_threads(cfg.intra_threads).map_err(|e| Error::Ort(e.into()))?;
	}
	if !dispatches.is_empty() {
		builder = builder.with_execution_providers(dispatches).map_err(|e| Error::Ort(e.into()))?;
	}
	Ok(builder.commit_from_file(model_path)?)
}

#[cfg(test)]
#[path = "tests/ep_tests.rs"]
mod tests;
