mod api;
mod dto;
mod error;
mod metrics;
mod profile;
mod state;

use std::{path::{Path, PathBuf}, sync::Arc};

use anyhow::Context;
use clap::{Parser, Subcommand};
use rsinfer_core::Registry;
use tokio::net::TcpListener;

use crate::state::AppState;

#[derive(Parser)]
#[command(name = "rsinfer", about = "ONNX Runtime inference server: embeddings, rerank, PII, zero-shot")]
struct Args {
	/// Path to the YAML model/server config (serve mode).
	#[arg(short, long, default_value = "config.yaml")]
	config: PathBuf,
	#[command(subcommand)]
	command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
	/// Download an ONNX model (+ tokenizer) from the Hugging Face Hub into a folder,
	/// so it can be referenced via `path:` in the server config without any downloading at startup.
	Download {
		/// HF repo id, e.g. Xenova/multilingual-e5-small
		repo: String,
		/// Destination folder; created if missing (e.g. models/e5-small).
		out: PathBuf,
		/// Repo revision: branch, tag, or commit hash.
		#[arg(long, default_value = "main")]
		revision: String,
		/// Pin an exact graph path in the repo, e.g. onnx/model_fp16.onnx
		/// (overrides the default model.onnx > fp16 > quantized preference).
		#[arg(long)]
		file: Option<String>,
		/// Subfolder inside the repo to also look in, e.g. onnx.
		#[arg(long)]
		subfolder: Option<String>,
		/// HF repo to take tokenizer.json / config.json from, for model-only ONNX repos.
		#[arg(long)]
		tokenizer_hf: Option<String>,
		/// Preferred weight format to look for in the repo: fp32 | fp16 | int8.
		#[arg(long)]
		dtype: Option<String>,
	},
	/// Micro-benchmark one configured model offline: synthetic batches through the real
	/// pipeline (session pool -> forward -> pooling), with optional ORT op-level profiling.
	Profile(profile::ProfileArgs),
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
	init_tracing();
	let args = Args::parse();
	match args.command {
		Some(Command::Download { repo, out, revision, file, subfolder, tokenizer_hf, dtype }) => {
			download(repo, out, revision, file, subfolder, tokenizer_hf, dtype).await
		}
		Some(Command::Profile(profile_args)) => profile::run(profile_args).await,
		None => serve(args.config).await,
	}
}

fn parse_dtype(s: &str) -> anyhow::Result<rsinfer_core::config::Dtype> {
	use rsinfer_core::config::Dtype;
	match s.to_ascii_lowercase().as_str() {
		"auto" => Ok(Dtype::Auto),
		"fp32" | "f32" => Ok(Dtype::Fp32),
		"fp16" | "f16" | "half" => Ok(Dtype::Fp16),
		"int8" | "q8" => Ok(Dtype::Int8),
		other => anyhow::bail!("unknown --dtype '{other}' (auto|fp32|fp16|int8)"),
	}
}

async fn download(repo: String, out: PathBuf, revision: String, file: Option<String>, subfolder: Option<String>, tokenizer_hf: Option<String>, dtype: Option<String>) -> anyhow::Result<()> {
	let cfg = rsinfer_core::ModelConfig {
		name: repo.clone(),
		hf: Some(repo.clone()),
		revision: revision.clone(),
		subfolder,
		file,
		tokenizer_hf,
		dtype: dtype.as_deref().map(parse_dtype).transpose()?.unwrap_or_default(),
		..Default::default()
	};
	let target = out.clone();
	tokio::task::spawn_blocking(move || rsinfer_core::hub::download_to(&cfg, &target, None))
		.await
		.map_err(|e| anyhow::anyhow!("download task failed: {e}"))?
		.map_err(|e| anyhow::anyhow!("{e}"))?;
	println!("downloaded {repo}@{revision} into {}", out.display());
	list_files(&out, &out)?;
	println!("\nuse it offline via a config entry:");
	let name = out.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| repo.rsplit('/').next().unwrap_or(&repo).to_string());
	println!("  - name: {name}\n    kind: embedding   # or rerank | pii | zeroshot\n    path: {}", out.display());
	Ok(())
}

fn list_files(root: &Path, dir: &Path) -> anyhow::Result<()> {
	let mut entries = Vec::new();
	collect_files(dir, &mut entries)?;
	entries.sort();
	for path in entries {
		let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
		let rel = path.strip_prefix(root).unwrap_or(&path);
		println!("  {:>9}  {}", human_size(size), rel.display());
	}
	Ok(())
}

fn collect_files(dir: &Path, out: &mut Vec<PathBuf>) -> anyhow::Result<()> {
	for e in std::fs::read_dir(dir).with_context(|| format!("cannot read {}", dir.display()))? {
		let path = e?.path();
		if path.is_dir() {
			collect_files(&path, out)?;
		} else {
			out.push(path);
		}
	}
	Ok(())
}

fn human_size(bytes: u64) -> String {
	const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
	let mut size = bytes as f64;
	let mut unit = 0;
	while size >= 1024.0 && unit < UNITS.len() - 1 {
		size /= 1024.0;
		unit += 1;
	}
	if unit == 0 { format!("{bytes} {}", UNITS[unit]) } else { format!("{size:.1} {}", UNITS[unit]) }
}

async fn serve(config_path: PathBuf) -> anyhow::Result<()> {
	let text = std::fs::read_to_string(&config_path).with_context(|| format!("cannot read config {}", config_path.display()))?;
	if config_path.extension().map(|e| e == "toml").unwrap_or(false) {
		anyhow::bail!("TOML configs are no longer supported; convert {} to YAML (see configs/config.example.yaml)", config_path.display());
	}
	let config: rsinfer_core::Config = serde_norway::from_str(&text).context("invalid YAML config")?;
	validate(&config)?;
	let config = Arc::new(config);

	tracing::info!(
		"rsinfer v{} starting with {} model(s) on {} ({}); EP features: coreml={} cuda={} tensorrt={} nvrtx={}",
		env!("CARGO_PKG_VERSION"),
		config.models.len(),
		std::env::consts::OS,
		std::env::consts::ARCH,
		cfg!(feature = "ep-coreml"),
		cfg!(feature = "ep-cuda"),
		cfg!(feature = "ep-tensorrt"),
		cfg!(feature = "ep-nvrtx"),
	);
	if let Some(rss) = rsinfer_core::memory::rss_mb() {
		tracing::info!(ram_mb = rss, "process memory at startup");
	}
	tracing::info!(bind = %config.server.bind, "resolving and loading configured models (first run downloads from the HF Hub)");
	let started = std::time::Instant::now();
	let registry = Arc::new(Registry::load(config.clone()).await?);
	registry.start_batchers();
	tracing::info!(
		elapsed_ms = started.elapsed().as_millis(),
		models = registry.infos().len(),
		ram_total_mb = ram_mb(),
		"all models loaded"
	);

	let state = AppState::new(registry, config.clone());
	let app = api::router(
		state,
		std::time::Duration::from_millis(config.server.request_timeout_ms),
		config.server.max_body_mb * 1024 * 1024,
	);

	let listener = TcpListener::bind(&config.server.bind).await.with_context(|| format!("cannot bind {}", config.server.bind))?;
	let addr = listener.local_addr()?;
	tracing::info!(
		%addr,
		url = %format!("http://{addr}"),
		total_startup_ms = started.elapsed().as_millis(),
		ram_total_mb = ram_mb(),
		"HTTP server ready; listening for requests"
	);
	axum::serve(listener, app)
		.with_graceful_shutdown(shutdown_signal())
		.await
		.context("server error")?;
	Ok(())
}

fn ram_mb() -> u64 {
	rsinfer_core::memory::rss_mb().unwrap_or(0)
}

fn validate(config: &rsinfer_core::Config) -> anyhow::Result<()> {
	let mut names = std::collections::HashSet::new();
	for m in &config.models {
		if !names.insert(m.name.clone()) {
			anyhow::bail!("duplicate model name '{}'", m.name);
		}
		m.validate().map_err(|e| anyhow::anyhow!("{e}"))?;
	}
	if config.models.is_empty() {
		anyhow::bail!("config has no models");
	}
	Ok(())
}

fn init_tracing() {
	use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};
	let explicit = std::env::var_os("RUST_LOG").is_some();
	let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info,ort=warn,tokenizers=warn,reqwest=warn,hf_hub=warn"));
	// Startup/lifecycle logs from our own crates are always visible, even if RUST_LOG is unset or looser.
	// An explicit RUST_LOG is authoritative: never raise the floor over it.
	let filter = if explicit {
		filter
	} else {
		["rsinfer", "rsinfer_server", "rsinfer_core"].into_iter().fold(filter, |f, t| {
			f.add_directive(format!("{t}=info").parse().expect("static directive parses"))
		})
	};
	tracing_subscriber::registry()
		.with(filter)
		.with(tracing_subscriber::fmt::layer().with_target(false))
		.init();
}

async fn shutdown_signal() {
	let ctrl_c = async {
		tokio::signal::ctrl_c().await.ok();
	};
	#[cfg(unix)]
	let sigterm = async {
		use tokio::signal::unix::{signal, SignalKind};
		if let Ok(mut s) = signal(SignalKind::terminate()) {
			s.recv().await;
		}
	};
	#[cfg(not(unix))]
	let sigterm = async { std::future::pending::<()>().await };
	tokio::select! {
		_ = ctrl_c => {},
		_ = sigterm => {},
	}
	tracing::info!("shutdown signal received");
}
