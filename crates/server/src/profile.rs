//! Offline micro-benchmark + ORT op-level profiling for one configured model.
//!
//! Runs synthetic token batches through the same pipeline the server uses
//! (session pool -> forward -> pooling -> normalize) and optionally dumps an
//! ONNX Runtime profiling report, aggregated per op type.

use std::{path::PathBuf, time::{Duration, Instant}};

use anyhow::{bail, Context};
use clap::Args;
use rsinfer_core::{config::ModelConfig, pipeline::embedding, registry, Kind, LoadedModel};

#[derive(Args)]
pub struct ProfileArgs {
	/// Path to the YAML config the model entry comes from.
	#[arg(long, default_value = "config.yaml")]
	pub config: PathBuf,
	/// Model name from the config; defaults to the first entry.
	#[arg(long)]
	pub model: Option<String>,
	/// Batch size (number of rows per forward).
	#[arg(long, default_value_t = 16)]
	pub batch: usize,
	/// Sequence length per row; 0 = the model's max_len (or 512).
	#[arg(long, default_value_t = 0)]
	pub seq: usize,
	/// Measured iterations.
	#[arg(long, default_value_t = 5)]
	pub iters: usize,
	/// Warmup iterations (excluded from timings).
	#[arg(long, default_value_t = 2)]
	pub warmup: usize,
	/// Dump an ORT profiling report here (prefix; writes <prefix>.json). Adds tracing overhead.
	#[arg(long)]
	pub profile_out: Option<PathBuf>,
	/// Override the model's intra-op thread count (0 = ORT default: all cores).
	#[arg(long)]
	pub intra_threads: Option<usize>,
}

pub async fn run(args: ProfileArgs) -> anyhow::Result<()> {
	if args.batch == 0 || args.iters == 0 {
		bail!("--batch and --iters must be >= 1");
	}
	let text = std::fs::read_to_string(&args.config).with_context(|| format!("cannot read {}", args.config.display()))?;
	let config: rsinfer_core::Config = serde_norway::from_str(&text).context("invalid YAML config")?;
	let mut cfg: ModelConfig = match &args.model {
		Some(n) => config.models.iter().find(|m| &m.name == n).cloned().with_context(|| format!("model '{n}' not in config"))?,
		None => config.models.first().cloned().context("config has no models")?,
	};
	if cfg.kind != Kind::Embedding {
		bail!("profile currently supports `kind: embedding` models (got '{}')", cfg.kind.as_str());
	}
	let seq = if args.seq > 0 { args.seq } else { cfg.max_len.unwrap_or(512) };
	if let Some(t) = args.intra_threads {
		cfg.intra_threads = t;
	}
	// Profiling is per-session; always measure one replica so every run hits the profiled session.
	cfg.replicas = 1;
	// Measure one forward of `--batch` rows, not `max_batch`-sized slices.
	cfg.max_batch = args.batch;
	cfg.profiling_prefix = args.profile_out.clone();

	let cache = config.server.hf_cache_dir.clone();
	let load_cfg = cfg.clone();
	let model = tokio::task::spawn_blocking(move || registry::load_model(&load_cfg, cache.as_deref(), usize::MAX))
		.await
		.map_err(|e| anyhow::anyhow!("loader task panicked: {e}"))??;

	println!(
		"model={} graph={} eps={} intra_threads={} batch={} seq={} warmup={} iters={}",
		model.name(),
		model.source,
		model.eps.join(","),
		if cfg.intra_threads > 0 { cfg.intra_threads.to_string() } else { "default".into() },
		args.batch,
		seq,
		args.warmup,
		args.iters,
	);

	let rows = synthetic_rows(&model, args.batch, seq);
	let queue_wait = Duration::from_secs(300);
	let mut timings = Vec::with_capacity(args.iters);
	let mut total_tokens = 0usize;
	for i in 0..(args.warmup + args.iters) {
		let t0 = Instant::now();
		let out = embedding::embed_tokens(&model, rows.clone(), queue_wait).await?;
		let ms = t0.elapsed().as_secs_f64() * 1e3;
		total_tokens = out.tokens;
		if i >= args.warmup {
			timings.push(ms);
			println!("  iter {:>2}  {:>9.1} ms", i - args.warmup + 1, ms);
		}
	}
	timings.sort_by(|a, b| a.partial_cmp(b).unwrap());
	let med = timings[timings.len() / 2];
	let padded = total_tokens;
	let label = if args.profile_out.is_some() { " (incl. ORT tracing overhead)" } else { "" };
	println!(
		"\nmin {:.0} ms | median {:.0} ms | max {:.0} ms | throughput {:.0} padded tok/s{}",
		timings[0],
		med,
		timings[timings.len() - 1],
		padded as f64 / (med / 1e3),
		label,
	);

	if let Some(prefix) = &args.profile_out {
		let mut pooled = model.pool.acquire(queue_wait).await?;
		let file = pooled.get_mut().end_profiling().map_err(|e| anyhow::anyhow!("end_profiling: {e}"))?;
		drop(pooled);
		let path = std::path::Path::new(&file);
		let resolved = if path.exists() { path.to_path_buf() } else { prefix.with_extension("json") };
		println!("\nORT profiling report: {}", resolved.display());
		print_op_summary(&resolved)?;
	}
	Ok(())
}

/// Deterministic pseudo-random token ids (no `rand` dependency needed).
fn synthetic_rows(model: &LoadedModel, batch: usize, seq: usize) -> Vec<Vec<u32>> {
	let vocab = model.encoder.vocab_size.saturating_sub(8).max(1000) as u32;
	let mut state = 0x853c49e6748fea9bu64;
	let mut next = move || {
		state ^= state << 13;
		state ^= state >> 7;
		state ^= state << 17;
		(state % u64::from(vocab - 4) + 4) as u32
	};
	(0..batch).map(|_| (0..seq).map(|_| next()).collect()).collect()
}

/// Aggregates an ORT profiling JSON (chrome-trace format) by op type and prints a table.
fn print_op_summary(path: &std::path::Path) -> anyhow::Result<()> {
	let raw = std::fs::read_to_string(path)
		.with_context(|| format!("cannot read profiling output {path:?} (profiling needs one more run before end_profiling)"))?;
	let doc: serde_json::Value = serde_json::from_str(&raw).context("profiling output is not valid JSON")?;
	// ORT emits either a bare array of trace events or a {"traceEvents": [...]} object.
	let events = match &doc {
		serde_json::Value::Array(a) => a.as_slice(),
		_ => doc.get("traceEvents").and_then(|v| v.as_array()).context("profiling output has no traceEvents").map(Vec::as_slice)?,
	};

	#[derive(Default)]
	struct Agg {
		count: u64,
		total_us: f64,
		provider: String,
	}
	let mut by_op: std::collections::HashMap<String, Agg> = std::collections::HashMap::new();
	for ev in events {
		if ev.get("ph").and_then(|v| v.as_str()) != Some("X") || ev.get("cat").and_then(|v| v.as_str()) != Some("Node") {
			continue;
		}
		let args = ev.get("args");
		let Some(dur) = ev.get("dur").and_then(|v| v.as_f64()).or_else(|| args.and_then(|a| a.get("execution_time_us")).and_then(|v| v.as_f64())) else { continue };
		let name = args
			.and_then(|a| a.get("op_name"))
			.and_then(|v| v.as_str())
			.or_else(|| ev.get("name").and_then(|v| v.as_str()))
			.unwrap_or("?")
			.to_string();
		let provider = args
			.and_then(|a| a.get("provider"))
			.and_then(|v| v.as_str())
			.unwrap_or("")
			.replace("ExecutionProvider", "");
		let agg = by_op.entry(name).or_default();
		agg.count += 1;
		agg.total_us += dur;
		if agg.provider.is_empty() {
			agg.provider = provider;
		}
	}
	let total: f64 = by_op.values().map(|a| a.total_us).sum();
	if total <= 0.0 {
		bail!("profiling report has no timed events (run at least one iteration with --profile-out)");
	}
	let mut rows: Vec<(String, Agg)> = by_op.into_iter().collect();
	rows.sort_by(|a, b| b.1.total_us.partial_cmp(&a.1.total_us).unwrap());
	println!("{:<34} {:>6} {:>10} {:>7}  provider", "op", "count", "total ms", "%");
	for (name, agg) in rows.iter().take(15) {
		println!("{:<34} {:>6} {:>10.1} {:>6.1}%  {}", name, agg.count, agg.total_us / 1e3, agg.total_us / total * 100.0, agg.provider);
	}
	println!("{:<34} {:>6} {:>10.1} {:>6.1}%", "TOTAL (node time, excl. gaps)", rows.iter().map(|r| r.1.count).sum::<u64>(), total / 1e3, 100.0);
	Ok(())
}
