//! Dynamic INT8 quantization of ONNX graphs, done by the server itself.
//!
//! Rewrites an fp32 graph the way `onnxruntime.quantization.quantize_dynamic`
//! does, but with per-channel weight scales (closer to fp32: on e5-small,
//! embedding cosine to fp32 0.997 vs 0.987 for the per-tensor int8 files
//! published on the Hub, at the same speed):
//! - `MatMul(A, W)` with a constant fp32 `W[K,N]` becomes
//!   `DynamicQuantizeLinear(A) -> MatMulInteger(A_q, W_q, A_zp, W_zp) -> Cast
//!   -> Mul(A_scale * W_scale)`, `W_q` int8 with one symmetric scale per output
//!   column; ONNX Runtime fuses this into its int8 kernels.
//! - `Gather(E, ids)` on a large constant table `E[V,D]` stores `E` as int8 with
//!   one scale per row, dequantized after the lookup: ~4x less memory for
//!   vocab-heavy models (multilingual / decoder embedding tables).
//!
//! Only the protobuf fields involved are decoded; everything else is copied
//! verbatim. Large tensors go to an external-data file next to the new graph.

use std::{
	collections::{HashMap, HashSet},
	fs::File,
	hash::{Hash, Hasher},
	io::{Read, Seek, SeekFrom, Write},
	path::{Component, Path, PathBuf},
};

use crate::Result;

/// Bump when the rewrite changes, so cached graphs are rebuilt.
const VERSION: u32 = 1;
/// Embedding tables smaller than this (elements) stay fp32: little to save.
const MIN_GATHER_ELEMS: usize = 1 << 16;
/// Tensors at least this large are written to the external-data file.
const EXTERNAL_MIN_BYTES: usize = 1024;
const DATA_FILE: &str = "model.onnx_data";

// ONNX TensorProto.DataType
const FLOAT: i64 = 1;
const INT8: i64 = 3;

/// What a rewrite changed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantStats {
	pub matmuls: usize,
	pub gathers: usize,
}

/// Returns the int8 rewrite of the graph at `src`, cached under `cache_root`
/// (built on first use), or `None` if it has nothing to quantize (already
/// quantized, fp16, opset < 11, no constant fp32 MatMul/Gather weights).
pub fn cached_int8(src: &Path, cache_root: &Path) -> Result<Option<PathBuf>> {
	let qmax = weight_qmax();
	let dir = cache_root.join("int8").join(cache_key(src, qmax)?);
	let graph = dir.join("model.onnx");
	if graph.is_file() {
		return Ok(Some(graph));
	}
	if dir.join("nothing-to-quantize").is_file() {
		return Ok(None);
	}
	let tmp = dir.with_extension(format!("tmp-{}", std::process::id()));
	let _ = std::fs::remove_dir_all(&tmp);
	std::fs::create_dir_all(&tmp)?;
	let stats = quantize_model(src, &tmp, qmax);
	let stats = match stats {
		Ok(s) => s,
		Err(e) => {
			let _ = std::fs::remove_dir_all(&tmp);
			return Err(e);
		}
	};
	if stats.is_none() {
		std::fs::write(tmp.join("nothing-to-quantize"), b"")?;
	}
	if let Err(e) = std::fs::rename(&tmp, &dir) {
		// Another process may have won the race; use its result.
		let _ = std::fs::remove_dir_all(&tmp);
		if !dir.is_dir() {
			return Err(e.into());
		}
	}
	Ok(graph.is_file().then_some(graph))
}

/// Cache key: source identity (path, size, mtime) + rewrite version + weight range.
fn cache_key(src: &Path, qmax: f32) -> Result<String> {
	let meta = std::fs::metadata(src)?;
	let mut h = std::collections::hash_map::DefaultHasher::new();
	std::fs::canonicalize(src).unwrap_or_else(|_| src.to_path_buf()).hash(&mut h);
	meta.len().hash(&mut h);
	meta.modified().ok().hash(&mut h);
	VERSION.hash(&mut h);
	qmax.to_bits().hash(&mut h);
	Ok(format!("{:016x}", h.finish()))
}

/// Weight range: 7-bit on x86 CPUs without VNNI, where ORT's u8 x s8 kernels
/// can saturate 16-bit intermediates (its `reduce_range` advice); 8-bit otherwise.
fn weight_qmax() -> f32 {
	#[cfg(target_arch = "x86_64")]
	{
		if !(std::arch::is_x86_feature_detected!("avxvnni") || std::arch::is_x86_feature_detected!("avx512vnni")) {
			return 63.0;
		}
	}
	127.0
}

/// Writes the int8 rewrite of `src` to `out_dir/model.onnx` (+ `model.onnx_data`).
/// `None` (and nothing written) when there is nothing to quantize.
pub fn quantize_model(src: &Path, out_dir: &Path, qmax: f32) -> Result<Option<QuantStats>> {
	let bytes = std::fs::read(src)?;
	let src_dir = src.parent().unwrap_or(Path::new("."));
	let top = pb::fields(&bytes)?;
	if default_opset(&top)? < 11 {
		return Ok(None); // DynamicQuantizeLinear needs opset 11
	}
	let graph_buf = top
		.iter()
		.find_map(|f| match (f.num, &f.value) {
			(7, pb::Value::Len(b)) => Some(*b),
			_ => None,
		})
		.ok_or_else(|| pb::bad("no graph"))?;
	let graph = pb::fields(graph_buf)?;

	let mut inits: HashMap<&str, Tensor> = HashMap::new();
	let mut graph_inputs: HashSet<&str> = HashSet::new();
	let mut nodes: Vec<Option<Node>> = Vec::with_capacity(graph.len());
	for f in &graph {
		nodes.push(None);
		match (f.num, &f.value) {
			(1, pb::Value::Len(b)) => *nodes.last_mut().expect("pushed") = Some(Node::parse(b)?),
			(5, pb::Value::Len(b)) => {
				let t = Tensor::parse(b)?;
				inits.insert(t.name, t);
			}
			(11, pb::Value::Len(b)) => {
				if let Some(name) = value_info_name(b)? {
					graph_inputs.insert(name);
				}
			}
			_ => {}
		}
	}

	// Pick nodes whose constant weight we can quantize.
	let is_default_domain = |d: &str| d.is_empty() || d == "ai.onnx";
	let fp32_2d = |name: &str| inits.get(name).filter(|t| t.data_type == FLOAT && t.dims.len() == 2 && t.dims.iter().all(|&d| d > 0));
	let mut targets: HashMap<usize, Kind> = HashMap::new();
	for (i, n) in nodes.iter().enumerate() {
		let Some(n) = n else { continue };
		if !is_default_domain(n.domain) || n.inputs.len() != 2 || n.outputs.len() != 1 {
			continue;
		}
		match n.op_type {
			"MatMul" if fp32_2d(n.inputs[1]).is_some() && !inits.contains_key(n.inputs[0]) => {
				targets.insert(i, Kind::MatMul);
			}
			"Gather"
				if n.axis_is_zero() && !inits.contains_key(n.inputs[1]) && fp32_2d(n.inputs[0]).is_some_and(|t| t.elems() >= MIN_GATHER_ELEMS) =>
			{
				targets.insert(i, Kind::Gather);
			}
			_ => {}
		}
	}
	if targets.is_empty() {
		return Ok(None);
	}
	// A weight is dropped only when every consumer is quantized and it is not a graph input.
	let mut uses: HashMap<&str, (usize, usize)> = HashMap::new(); // (all uses, quantized uses)
	for (i, n) in nodes.iter().enumerate() {
		let Some(n) = n else { continue };
		for (k, input) in n.inputs.iter().enumerate() {
			let e = uses.entry(input).or_default();
			e.0 += 1;
			let weight_slot = match targets.get(&i) {
				Some(Kind::MatMul) => 1,
				Some(Kind::Gather) => 0,
				None => usize::MAX,
			};
			if k == weight_slot {
				e.1 += 1;
			}
		}
	}
	let dropped: HashSet<&str> = targets
		.iter()
		.map(|(&i, kind)| {
			let n = nodes[i].as_ref().expect("target is a node");
			n.inputs[if *kind == Kind::MatMul { 1 } else { 0 }]
		})
		.filter(|w| !graph_inputs.contains(w) && uses.get(w).is_some_and(|&(all, q)| all == q))
		.collect();

	let mut data = DataFile::new(out_dir.join(DATA_FILE));
	let mut new_inits: Vec<u8> = Vec::new();
	let mut quantized_weights: HashSet<&str> = HashSet::new();
	let mut quantized_acts: HashSet<&str> = HashSet::new();
	let mut stats = QuantStats { matmuls: 0, gathers: 0 };
	let mut g: Vec<u8> = Vec::with_capacity(graph_buf.len());
	for (i, f) in graph.iter().enumerate() {
		if let Some(kind) = targets.get(&i) {
			let n = nodes[i].as_ref().expect("target is a node");
			match kind {
				Kind::MatMul => {
					let (a, w, y) = (n.inputs[0], n.inputs[1], n.outputs[0]);
					if quantized_weights.insert(w) {
						let t = &inits[w];
						let values = t.load_f32(src_dir)?;
						let (q, scales) = quantize_columns(&values, t.dims[0] as usize, t.dims[1] as usize, qmax);
						put_tensor(&mut new_inits, &format!("{w}_rsq_int8"), &t.dims, INT8, &as_bytes_i8(&q), &mut data)?;
						put_tensor(&mut new_inits, &format!("{w}_rsq_scale"), &[t.dims[1]], FLOAT, &as_bytes_f32(&scales), &mut data)?;
						put_tensor(&mut new_inits, &format!("{w}_rsq_zp"), &[t.dims[1]], INT8, &vec![0u8; scales.len()], &mut data)?;
					}
					if quantized_acts.insert(a) {
						let outs = [format!("{a}_rsq_q"), format!("{a}_rsq_scale"), format!("{a}_rsq_zp")];
						put_node(&mut g, "DynamicQuantizeLinear", &[a], &[&outs[0], &outs[1], &outs[2]], None);
					}
					let (aq, ascale, azp) = (format!("{a}_rsq_q"), format!("{a}_rsq_scale"), format!("{a}_rsq_zp"));
					let (yi, yf, ys) = (format!("{y}_rsq_int32"), format!("{y}_rsq_f32"), format!("{y}_rsq_scale"));
					put_node(&mut g, "MatMulInteger", &[&aq, &format!("{w}_rsq_int8"), &azp, &format!("{w}_rsq_zp")], &[&yi], None);
					put_node(&mut g, "Cast", &[&yi], &[&yf], Some(("to", FLOAT)));
					put_node(&mut g, "Mul", &[&ascale, &format!("{w}_rsq_scale")], &[&ys], None);
					put_node(&mut g, "Mul", &[&yf, &ys], &[y], None);
					stats.matmuls += 1;
				}
				Kind::Gather => {
					let (e, ids, y) = (n.inputs[0], n.inputs[1], n.outputs[0]);
					if quantized_weights.insert(e) {
						let t = &inits[e];
						let values = t.load_f32(src_dir)?;
						let (q, scales) = quantize_rows(&values, t.dims[0] as usize, t.dims[1] as usize);
						put_tensor(&mut new_inits, &format!("{e}_rsq_int8"), &t.dims, INT8, &as_bytes_i8(&q), &mut data)?;
						put_tensor(&mut new_inits, &format!("{e}_rsq_rowscale"), &[t.dims[0], 1], FLOAT, &as_bytes_f32(&scales), &mut data)?;
					}
					let (yq, yf, ys) = (format!("{y}_rsq_int8"), format!("{y}_rsq_f32"), format!("{y}_rsq_rowscale"));
					put_node(&mut g, "Gather", &[&format!("{e}_rsq_int8"), ids], &[&yq], None);
					put_node(&mut g, "Cast", &[&yq], &[&yf], Some(("to", FLOAT)));
					put_node(&mut g, "Gather", &[&format!("{e}_rsq_rowscale"), ids], &[&ys], None);
					put_node(&mut g, "Mul", &[&yf, &ys], &[y], None);
					stats.gathers += 1;
				}
			}
			continue;
		}
		if let (5, pb::Value::Len(b)) = (f.num, &f.value) {
			let t = Tensor::parse(b)?;
			if dropped.contains(t.name) {
				continue;
			}
			if t.external.is_some() {
				// Re-home external data into our own file (locations are relative to the graph).
				put_tensor(&mut g, t.name, &t.dims, t.data_type, &t.load_bytes(src_dir)?, &mut data)?;
				continue;
			}
		}
		g.extend_from_slice(f.raw);
	}
	g.extend_from_slice(&new_inits);

	let mut model = Vec::with_capacity(g.len() + 1024);
	for f in &top {
		if f.num == 7 {
			pb::put_len(&mut model, 7, &g);
		} else {
			model.extend_from_slice(f.raw);
		}
	}
	data.finish()?;
	std::fs::write(out_dir.join("model.onnx"), model)?;
	Ok(Some(stats))
}

/// Sentences the quality gate runs through both graphs: mixed EN/FR, with
/// names, contacts and account numbers so PII models see entities.
const CALIBRATION: &[&str] = &[
	"The maintenance manual describes the inspection interval for the landing gear actuators.",
	"Le rapport d'essai indique une consommation de carburant inférieure aux prévisions.",
	"Please contact Maria Gonzalez at maria.gonzalez@example.com or +33 6 12 34 56 78.",
	"Wire the payment to IBAN FR76 3000 6000 0112 3456 7890 189 before Friday.",
	"Engineers reviewed the wiring harness routing to reduce electromagnetic interference.",
	"Jean Dupont habite 12 rue des Lilas, 31000 Toulouse, depuis 2019.",
	"Quarterly revenue grew by twelve percent thanks to strong aftermarket services.",
	"How do I reset my password on the corporate travel portal?",
	"The supplier delivered the composite panels two weeks ahead of schedule.",
	"Les données de vol montrent une réduction du bruit en phase d'approche.",
	"Patient John Smith, born 03/04/1985, was admitted on March 2nd.",
	"A root cause analysis was opened after the hydraulic pressure warning during taxi.",
];

/// Calibration batch for a model kind: single texts, or (text, text) pairs for
/// cross-encoders (mostly unrelated pairs plus a few identical ones, so both
/// low and high scores are compared).
pub(crate) fn calibration(encoder: &crate::tokenize::Encoder, kind: crate::Kind) -> Result<crate::tokenize::Encoded> {
	let texts: Vec<String> = CALIBRATION.iter().map(|s| s.to_string()).collect();
	match kind {
		crate::Kind::Embedding | crate::Kind::Pii => encoder.encode_texts(&texts),
		crate::Kind::Rerank | crate::Kind::Zeroshot => {
			let n = texts.len();
			let pairs: Vec<(String, String)> = (0..n)
				.map(|i| (texts[i].clone(), texts[(i * 5 + 3) % n].clone()))
				.chain(texts.iter().take(4).map(|t| (t.clone(), t.clone())))
				.collect();
			encoder.encode_pairs(&pairs)
		}
	}
}

/// Outcome of comparing the published graph's outputs with the int8 rewrite's.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Check {
	pub metric: &'static str,
	/// Worst row.
	pub worst: f64,
	pub threshold: f64,
	pub pass: bool,
}

/// Worst-row agreement between reference and int8 outputs: cosine for vectors
/// and logits (>= 0.99), 1 - |delta| for scores (>= 0.95), argmax label
/// agreement over real tokens for token classification (>= 0.97).
pub(crate) fn compare(reference: &[crate::pipeline::RowOut], candidate: &[crate::pipeline::RowOut], mask: &[Vec<i64>]) -> Result<Check> {
	use crate::pipeline::RowOut;
	if reference.len() != candidate.len() || reference.is_empty() {
		return Err(pb::bad("quality check produced mismatched outputs"));
	}
	let mut check = Check { metric: "", worst: f64::INFINITY, threshold: 0.0, pass: false };
	for (i, (a, b)) in reference.iter().zip(candidate).enumerate() {
		let (metric, value, threshold) = match (a, b) {
			(RowOut::Vector(x), RowOut::Vector(y)) | (RowOut::Logits(x), RowOut::Logits(y)) => ("cosine", cosine(x, y), 0.99),
			(RowOut::Score(x), RowOut::Score(y)) => ("score agreement", 1.0 - (x - y).abs(), 0.95),
			(RowOut::Tokens(x), RowOut::Tokens(y)) => {
				let real: Vec<usize> = mask.get(i).map(|m| (0..m.len()).filter(|&j| m[j] != 0).collect()).unwrap_or_default();
				let agree = real.iter().filter(|&&j| x.get(j).map(|t| t.0) == y.get(j).map(|t| t.0)).count();
				("token label agreement", agree as f64 / real.len().max(1) as f64, 0.97)
			}
			_ => return Err(pb::bad("quality check produced different output kinds")),
		};
		check.metric = metric;
		check.threshold = threshold;
		check.worst = check.worst.min(value);
	}
	check.pass = check.worst >= check.threshold;
	Ok(check)
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
	let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
	for (&x, &y) in a.iter().zip(b) {
		let (x, y) = (f64::from(x), f64::from(y));
		dot += x * y;
		na += x * x;
		nb += y * y;
	}
	if na == 0.0 && nb == 0.0 {
		return 1.0;
	}
	dot / (na.sqrt() * nb.sqrt()).max(f64::MIN_POSITIVE)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
	MatMul,
	Gather,
}

/// Per-column symmetric int8 for `W[K,N]` (row-major): scale_n = max|W[:,n]| / qmax.
fn quantize_columns(w: &[f32], k: usize, n: usize, qmax: f32) -> (Vec<i8>, Vec<f32>) {
	let mut scales = vec![0f32; n];
	for row in w.chunks_exact(n).take(k) {
		for (s, &v) in scales.iter_mut().zip(row) {
			*s = s.max(v.abs());
		}
	}
	scales.iter_mut().for_each(|s| *s = if *s > 0.0 { *s / qmax } else { 1.0 });
	let q = w.chunks_exact(n).flat_map(|row| row.iter().zip(&scales).map(|(&v, &s)| (v / s).round().clamp(-qmax, qmax) as i8)).collect();
	(q, scales)
}

/// Per-row symmetric int8 for an embedding table `E[V,D]`.
fn quantize_rows(e: &[f32], v: usize, d: usize) -> (Vec<i8>, Vec<f32>) {
	let mut q = Vec::with_capacity(v * d);
	let mut scales = Vec::with_capacity(v);
	for row in e.chunks_exact(d).take(v) {
		let max = row.iter().fold(0f32, |m, x| m.max(x.abs()));
		let s = if max > 0.0 { max / 127.0 } else { 1.0 };
		q.extend(row.iter().map(|&x| (x / s).round().clamp(-127.0, 127.0) as i8));
		scales.push(s);
	}
	(q, scales)
}

fn as_bytes_i8(v: &[i8]) -> Vec<u8> {
	v.iter().map(|&x| x as u8).collect()
}

fn as_bytes_f32(v: &[f32]) -> Vec<u8> {
	v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// The `ai.onnx` opset version the model imports (0 if none).
fn default_opset(top: &[pb::Field<'_>]) -> Result<i64> {
	let mut version = 0;
	for f in top.iter().filter(|f| f.num == 8) {
		let pb::Value::Len(b) = f.value else { continue };
		let (mut domain, mut v) = ("", 0);
		for g in pb::fields(b)? {
			match (g.num, g.value) {
				(1, pb::Value::Len(s)) => domain = pb::str(s)?,
				(2, pb::Value::Varint(x)) => v = x as i64,
				_ => {}
			}
		}
		if domain.is_empty() || domain == "ai.onnx" {
			version = version.max(v);
		}
	}
	Ok(version)
}

fn value_info_name(buf: &[u8]) -> Result<Option<&str>> {
	for f in pb::fields(buf)? {
		if let (1, pb::Value::Len(s)) = (f.num, f.value) {
			return Ok(Some(pb::str(s)?));
		}
	}
	Ok(None)
}

/// The NodeProto fields the rewrite needs.
struct Node<'a> {
	inputs: Vec<&'a str>,
	outputs: Vec<&'a str>,
	op_type: &'a str,
	domain: &'a str,
	/// Integer attributes by name (others are only counted).
	int_attrs: Vec<(&'a str, i64)>,
	other_attrs: usize,
}

impl<'a> Node<'a> {
	fn parse(buf: &'a [u8]) -> Result<Self> {
		let mut n = Node { inputs: Vec::new(), outputs: Vec::new(), op_type: "", domain: "", int_attrs: Vec::new(), other_attrs: 0 };
		for f in pb::fields(buf)? {
			match (f.num, f.value) {
				(1, pb::Value::Len(s)) => n.inputs.push(pb::str(s)?),
				(2, pb::Value::Len(s)) => n.outputs.push(pb::str(s)?),
				(4, pb::Value::Len(s)) => n.op_type = pb::str(s)?,
				(7, pb::Value::Len(s)) => n.domain = pb::str(s)?,
				(5, pb::Value::Len(a)) => {
					let (mut name, mut int) = ("", None);
					for g in pb::fields(a)? {
						match (g.num, g.value) {
							(1, pb::Value::Len(s)) => name = pb::str(s)?,
							(3, pb::Value::Varint(x)) => int = Some(x as i64),
							_ => {}
						}
					}
					match int {
						Some(v) => n.int_attrs.push((name, v)),
						None => n.other_attrs += 1,
					}
				}
				_ => {}
			}
		}
		Ok(n)
	}

	/// No attributes other than an optional `axis = 0`.
	fn axis_is_zero(&self) -> bool {
		self.other_attrs == 0 && self.int_attrs.iter().all(|&(name, v)| name == "axis" && v == 0)
	}
}

/// External-data reference of a TensorProto.
struct External {
	location: String,
	offset: u64,
	length: Option<u64>,
}

/// The TensorProto fields the rewrite needs.
struct Tensor<'a> {
	name: &'a str,
	dims: Vec<i64>,
	data_type: i64,
	raw: Option<&'a [u8]>,
	float_data: Vec<f32>,
	external: Option<External>,
}

impl<'a> Tensor<'a> {
	fn parse(buf: &'a [u8]) -> Result<Self> {
		let mut t = Tensor { name: "", dims: Vec::new(), data_type: 0, raw: None, float_data: Vec::new(), external: None };
		let mut ext: Vec<(&str, &str)> = Vec::new();
		let mut location_external = false;
		for f in pb::fields(buf)? {
			match (f.num, f.value) {
				(1, pb::Value::Varint(d)) => t.dims.push(d as i64),
				(1, pb::Value::Len(packed)) => {
					let mut pos = 0;
					while pos < packed.len() {
						t.dims.push(pb::varint(packed, &mut pos)? as i64);
					}
				}
				(2, pb::Value::Varint(d)) => t.data_type = d as i64,
				(8, pb::Value::Len(s)) => t.name = pb::str(s)?,
				(9, pb::Value::Len(b)) => t.raw = Some(b),
				(4, pb::Value::I32(b)) => t.float_data.push(f32::from_le_bytes(b.try_into().expect("4 bytes"))),
				(4, pb::Value::Len(b)) => t.float_data.extend(b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().expect("4 bytes")))),
				(13, pb::Value::Len(entry)) => {
					let (mut k, mut v) = ("", "");
					for g in pb::fields(entry)? {
						match (g.num, g.value) {
							(1, pb::Value::Len(s)) => k = pb::str(s)?,
							(2, pb::Value::Len(s)) => v = pb::str(s)?,
							_ => {}
						}
					}
					ext.push((k, v));
				}
				(14, pb::Value::Varint(1)) => location_external = true,
				_ => {}
			}
		}
		if location_external {
			let get = |key: &str| ext.iter().find(|(k, _)| *k == key).map(|(_, v)| *v);
			let num = |key: &str| -> Result<Option<u64>> { get(key).map(|v| v.parse::<u64>().map_err(|_| pb::bad("bad external offset/length"))).transpose() };
			t.external = Some(External {
				location: get("location").ok_or_else(|| pb::bad("external tensor without location"))?.to_string(),
				offset: num("offset")?.unwrap_or(0),
				length: num("length")?,
			});
		}
		Ok(t)
	}

	fn elems(&self) -> usize {
		self.dims.iter().map(|&d| d.max(0) as usize).product()
	}

	/// Raw little-endian bytes of the tensor data, wherever it is stored.
	fn load_bytes(&self, src_dir: &Path) -> Result<Vec<u8>> {
		if let Some(ext) = &self.external {
			let rel = Path::new(&ext.location);
			if rel.is_absolute() || rel.components().any(|c| matches!(c, Component::ParentDir | Component::Prefix(_))) {
				return Err(pb::bad("external data location escapes the model directory"));
			}
			let mut file = File::open(src_dir.join(rel))?;
			let len = match ext.length {
				Some(l) => l,
				None => file.metadata()?.len().saturating_sub(ext.offset),
			};
			file.seek(SeekFrom::Start(ext.offset))?;
			let mut buf = vec![0u8; len as usize];
			file.read_exact(&mut buf)?;
			return Ok(buf);
		}
		if let Some(raw) = self.raw {
			return Ok(raw.to_vec());
		}
		Ok(as_bytes_f32(&self.float_data))
	}

	fn load_f32(&self, src_dir: &Path) -> Result<Vec<f32>> {
		let values: Vec<f32> = if self.external.is_none() && self.raw.is_none() {
			self.float_data.clone()
		} else {
			self.load_bytes(src_dir)?.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().expect("4 bytes"))).collect()
		};
		if values.len() != self.elems() {
			return Err(pb::bad(&format!("tensor '{}' has {} values for shape {:?}", self.name, values.len(), self.dims)));
		}
		Ok(values)
	}
}

/// Appends a NodeProto (graph field 1).
fn put_node(out: &mut Vec<u8>, op_type: &str, inputs: &[&str], outputs: &[&str], int_attr: Option<(&str, i64)>) {
	let mut n = Vec::new();
	for i in inputs {
		pb::put_len(&mut n, 1, i.as_bytes());
	}
	for o in outputs {
		pb::put_len(&mut n, 2, o.as_bytes());
	}
	pb::put_len(&mut n, 3, format!("{}_rsq_{op_type}", outputs[0]).as_bytes());
	pb::put_len(&mut n, 4, op_type.as_bytes());
	if let Some((name, v)) = int_attr {
		let mut a = Vec::new();
		pb::put_len(&mut a, 1, name.as_bytes());
		pb::put_uint(&mut a, 3, v as u64);
		pb::put_uint(&mut a, 20, 2); // AttributeType INT
		pb::put_len(&mut n, 5, &a);
	}
	pb::put_len(out, 1, &n);
}

/// Appends an initializer (graph field 5), externalizing large payloads.
fn put_tensor(out: &mut Vec<u8>, name: &str, dims: &[i64], data_type: i64, payload: &[u8], data: &mut DataFile) -> Result<()> {
	let mut t = Vec::new();
	for &d in dims {
		pb::put_uint(&mut t, 1, d as u64);
	}
	pb::put_uint(&mut t, 2, data_type as u64);
	pb::put_len(&mut t, 8, name.as_bytes());
	if payload.len() >= EXTERNAL_MIN_BYTES {
		let offset = data.append(payload)?;
		for (k, v) in [("location", DATA_FILE.to_string()), ("offset", offset.to_string()), ("length", payload.len().to_string())] {
			let mut e = Vec::new();
			pb::put_len(&mut e, 1, k.as_bytes());
			pb::put_len(&mut e, 2, v.as_bytes());
			pb::put_len(&mut t, 13, &e);
		}
		pb::put_uint(&mut t, 14, 1); // DataLocation EXTERNAL
	} else {
		pb::put_len(&mut t, 9, payload);
	}
	pb::put_len(out, 5, &t);
	Ok(())
}

/// The external-data file of the rewritten graph, created on first write.
struct DataFile {
	path: PathBuf,
	file: Option<std::io::BufWriter<File>>,
	len: u64,
}

impl DataFile {
	fn new(path: PathBuf) -> Self {
		Self { path, file: None, len: 0 }
	}

	/// Appends 64-byte-aligned and returns the offset.
	fn append(&mut self, payload: &[u8]) -> Result<u64> {
		if self.file.is_none() {
			self.file = Some(std::io::BufWriter::new(File::create(&self.path)?));
		}
		let file = self.file.as_mut().expect("just created");
		let pad = (64 - self.len % 64) % 64;
		file.write_all(&vec![0u8; pad as usize])?;
		let offset = self.len + pad;
		file.write_all(payload)?;
		self.len = offset + payload.len() as u64;
		Ok(offset)
	}

	fn finish(self) -> Result<()> {
		if let Some(mut f) = self.file {
			f.flush()?;
		}
		Ok(())
	}
}

/// Minimal protobuf wire-format reader/writer.
pub(crate) mod pb {
	use crate::{Error, Result};

	#[derive(Clone, Copy)]
	pub(crate) enum Value<'a> {
		Varint(u64),
		I64(#[allow(dead_code)] &'a [u8]),
		Len(&'a [u8]),
		I32(&'a [u8]),
	}

	pub(crate) struct Field<'a> {
		pub num: u32,
		pub value: Value<'a>,
		/// The whole field (key included), for verbatim copies.
		pub raw: &'a [u8],
	}

	pub(crate) fn bad(msg: &str) -> Error {
		Error::Config(format!("cannot quantize ONNX model: {msg}"))
	}

	pub(crate) fn varint(buf: &[u8], pos: &mut usize) -> Result<u64> {
		let mut v = 0u64;
		let mut shift = 0;
		loop {
			let b = *buf.get(*pos).ok_or_else(|| bad("truncated varint"))?;
			*pos += 1;
			v |= u64::from(b & 0x7f) << shift;
			if b & 0x80 == 0 {
				return Ok(v);
			}
			shift += 7;
			if shift >= 64 {
				return Err(bad("varint too long"));
			}
		}
	}

	fn take<'a>(buf: &'a [u8], pos: &mut usize, n: usize) -> Result<&'a [u8]> {
		let end = pos.checked_add(n).filter(|&e| e <= buf.len()).ok_or_else(|| bad("truncated field"))?;
		let s = &buf[*pos..end];
		*pos = end;
		Ok(s)
	}

	pub(crate) fn fields(buf: &[u8]) -> Result<Vec<Field<'_>>> {
		let mut out = Vec::new();
		let mut pos = 0;
		while pos < buf.len() {
			let start = pos;
			let key = varint(buf, &mut pos)?;
			let value = match key & 7 {
				0 => Value::Varint(varint(buf, &mut pos)?),
				1 => Value::I64(take(buf, &mut pos, 8)?),
				2 => {
					let n = usize::try_from(varint(buf, &mut pos)?).map_err(|_| bad("field too large"))?;
					Value::Len(take(buf, &mut pos, n)?)
				}
				5 => Value::I32(take(buf, &mut pos, 4)?),
				w => return Err(bad(&format!("unsupported wire type {w}"))),
			};
			out.push(Field { num: (key >> 3) as u32, value, raw: &buf[start..pos] });
		}
		Ok(out)
	}

	pub(crate) fn str(b: &[u8]) -> Result<&str> {
		std::str::from_utf8(b).map_err(|_| bad("non-UTF-8 string"))
	}

	pub(crate) fn put_varint(out: &mut Vec<u8>, mut v: u64) {
		while v >= 0x80 {
			out.push((v as u8) | 0x80);
			v >>= 7;
		}
		out.push(v as u8);
	}

	pub(crate) fn put_len(out: &mut Vec<u8>, num: u32, data: &[u8]) {
		put_varint(out, (u64::from(num) << 3) | 2);
		put_varint(out, data.len() as u64);
		out.extend_from_slice(data);
	}

	pub(crate) fn put_uint(out: &mut Vec<u8>, num: u32, v: u64) {
		put_varint(out, u64::from(num) << 3);
		put_varint(out, v);
	}
}

#[cfg(test)]
#[path = "tests/quantize_tests.rs"]
mod tests;
