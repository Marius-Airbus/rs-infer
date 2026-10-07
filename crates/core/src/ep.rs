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
pub const ALL_EPS: &[EpName] = &[EpName::Cpu, EpName::Coreml, EpName::Cuda, EpName::Tensorrt, EpName::Nvrtx];

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
		_ => false,
	}
}

#[allow(unused_variables)]
fn dispatch(cfg: &ModelConfig, name: EpName) -> Option<ExecutionProviderDispatch> {
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
		EpName::Cuda => Some(ep::CUDA::default().with_device_id(cfg.device_id).build()),
		#[cfg(feature = "ep-tensorrt")]
		EpName::Tensorrt => {
			let mut b = ep::TensorRT::default().with_device_id(cfg.device_id).with_engine_cache(true);
			if let Some(dir) = &cfg.trt_engine_cache {
				b = b.with_engine_cache_path(dir.display().to_string());
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
		_ => None,
	}
}

/// Resolves the effective EP list (request order, compiled-in + runtime-available, CPU appended)
/// together with the registrations that can actually be attempted.
pub fn resolve_eps(cfg: &ModelConfig) -> (Vec<String>, Vec<ExecutionProviderDispatch>) {
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
		match dispatch(cfg, name) {
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
	let (used, dispatches) = resolve_eps(cfg);
	tracing::debug!(model = cfg.name.as_str(), eps = ?used, "building session");

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
