//! Pooling inside the ONNX graph, for embedding exports that only output token
//! states (`last_hidden_state [B,T,D]`, the usual Xenova / onnx-community layout).
//!
//! Without it, every forward hands the whole token tensor back for the server to
//! pool on the CPU: on a GPU that is a device-to-host copy T times larger than
//! the `[B,D]` result (64 x 512 x 1024 fp32 = 128 MB per batch). Appending the
//! pooling to the graph makes the runtime compute and return only `[B,D]`, as
//! the new output [`POOLED_OUTPUT`]. External weight files are hard-linked, not copied.

use std::{
	hash::{Hash, Hasher},
	path::{Component, Path, PathBuf},
};

use crate::{
	config::Pooling,
	quantize::{default_opset, external_locations, pb, value_info_name},
	Result,
};

/// Name of the pooled `[B,D]` output added to the graph.
pub const POOLED_OUTPUT: &str = "rsinfer_pooled";
/// Bump when the rewrite changes, so cached graphs are rebuilt.
const VERSION: u32 = 1;

// ONNX TensorProto.DataType
const FLOAT: u64 = 1;
const INT64: u64 = 7;

/// Graph output names of the model at `path`.
pub fn output_names(path: &Path) -> Result<Vec<String>> {
	let bytes = std::fs::read(path)?;
	let graph = graph_fields(&bytes)?;
	let mut out = Vec::new();
	for f in pb::fields(graph)? {
		if let (12, pb::Value::Len(b)) = (f.num, f.value) {
			if let Some(name) = value_info_name(b)? {
				out.push(name.to_string());
			}
		}
	}
	Ok(out)
}

/// The graph at `src` with `pooling` of `token_output` appended as
/// [`POOLED_OUTPUT`], cached under `cache_root` (built on first use).
pub fn cached_pooled(src: &Path, token_output: &str, pooling: Pooling, cache_root: &Path) -> Result<PathBuf> {
	let meta = std::fs::metadata(src)?;
	let mut h = std::collections::hash_map::DefaultHasher::new();
	std::fs::canonicalize(src).unwrap_or_else(|_| src.to_path_buf()).hash(&mut h);
	meta.len().hash(&mut h);
	meta.modified().ok().hash(&mut h);
	(VERSION, token_output, format!("{pooling:?}")).hash(&mut h);
	let dir = cache_root.join("pooled").join(format!("{:016x}", h.finish()));
	let graph = dir.join("model.onnx");
	if graph.is_file() {
		return Ok(graph);
	}
	let tmp = dir.with_extension(format!("tmp-{}", std::process::id()));
	let _ = std::fs::remove_dir_all(&tmp);
	std::fs::create_dir_all(&tmp)?;
	if let Err(e) = rewrite(src, &tmp, token_output, pooling) {
		let _ = std::fs::remove_dir_all(&tmp);
		return Err(e);
	}
	if let Err(e) = std::fs::rename(&tmp, &dir) {
		let _ = std::fs::remove_dir_all(&tmp);
		if !graph.is_file() {
			return Err(e.into());
		}
	}
	Ok(graph)
}

fn graph_fields(model: &[u8]) -> Result<&[u8]> {
	pb::fields(model)?
		.into_iter()
		.find_map(|f| match (f.num, f.value) {
			(7, pb::Value::Len(b)) => Some(b),
			_ => None,
		})
		.ok_or_else(|| pb::bad("no graph"))
}

/// Writes `out_dir/model.onnx`: the source graph verbatim plus the pooling nodes.
fn rewrite(src: &Path, out_dir: &Path, token_output: &str, pooling: Pooling) -> Result<()> {
	let bytes = std::fs::read(src)?;
	let top = pb::fields(&bytes)?;
	let opset = default_opset(&top)?;
	// Negative Unsqueeze axes need opset 11; GatherND's `batch_dims` (last-token pooling) 12.
	let needed = if pooling == Pooling::Last { 12 } else { 11 };
	if opset < needed {
		return Err(pb::bad(&format!("{pooling:?} pooling in the graph needs opset >= {needed}, the model has {opset}")));
	}
	let graph_buf = graph_fields(&bytes)?;
	let graph = pb::fields(graph_buf)?;
	let names = |num: u32| -> Result<Vec<&str>> {
		let mut v = Vec::new();
		for f in &graph {
			if let (n, pb::Value::Len(b)) = (f.num, f.value) {
				if n == num {
					v.extend(value_info_name(b)?);
				}
			}
		}
		Ok(v)
	};
	if !names(11)?.contains(&"attention_mask") {
		return Err(pb::bad("no attention_mask input to pool with"));
	}
	if !names(12)?.contains(&token_output) {
		return Err(pb::bad(&format!("no '{token_output}' output to pool")));
	}

	let mut g = graph_buf.to_vec();
	let axes_as_input = opset >= 13; // Unsqueeze / ReduceSum moved `axes` from attribute to input
	let mut nodes = Nodes { g: &mut g, axes_as_input };
	match pooling {
		Pooling::Mean | Pooling::Auto => {
			nodes.cast("attention_mask", "rsp_mask_f", FLOAT);
			nodes.with_axes("Unsqueeze", "rsp_mask_f", &[-1], None, "rsp_mask3");
			nodes.node("Mul", &[token_output, "rsp_mask3"], "rsp_masked", &[]);
			nodes.with_axes("ReduceSum", "rsp_masked", &[1], Some(0), "rsp_sum");
			nodes.with_axes("ReduceSum", "rsp_mask3", &[1], Some(0), "rsp_count");
			put_scalar(nodes.g, "rsp_eps", FLOAT, &1e-9f32.to_le_bytes());
			nodes.node("Max", &["rsp_count", "rsp_eps"], "rsp_count_safe", &[]);
			nodes.node("Div", &["rsp_sum", "rsp_count_safe"], POOLED_OUTPUT, &[]);
		}
		Pooling::Cls => {
			put_scalar(nodes.g, "rsp_zero", INT64, &0i64.to_le_bytes());
			nodes.node("Gather", &[token_output, "rsp_zero"], POOLED_OUTPUT, &[("axis", 1)]);
		}
		Pooling::Last => {
			// Rows are right-padded: the last real token sits at (number of real tokens - 1).
			nodes.cast("attention_mask", "rsp_mask_i64", INT64);
			nodes.with_axes("ReduceSum", "rsp_mask_i64", &[1], Some(1), "rsp_len");
			put_scalar(nodes.g, "rsp_one", INT64, &1i64.to_le_bytes());
			nodes.node("Sub", &["rsp_len", "rsp_one"], "rsp_last", &[]);
			nodes.node("GatherND", &[token_output, "rsp_last"], POOLED_OUTPUT, &[("batch_dims", 1)]);
		}
	}
	// Graph output (ValueInfoProto: name + float tensor type, shape left open).
	let mut tensor_type = Vec::new();
	pb::put_uint(&mut tensor_type, 1, FLOAT);
	let mut type_proto = Vec::new();
	pb::put_len(&mut type_proto, 1, &tensor_type);
	let mut output = Vec::new();
	pb::put_len(&mut output, 1, POOLED_OUTPUT.as_bytes());
	pb::put_len(&mut output, 2, &type_proto);
	pb::put_len(&mut g, 12, &output);

	let mut model = Vec::with_capacity(g.len() + 256);
	for f in &top {
		if f.num == 7 {
			pb::put_len(&mut model, 7, &g);
		} else {
			model.extend_from_slice(f.raw);
		}
	}
	let src_dir = src.parent().unwrap_or(Path::new("."));
	for location in external_locations(&graph)? {
		link_external(&src_dir.join(&location), &out_dir.join(&location), &location)?;
	}
	std::fs::write(out_dir.join("model.onnx"), model)?;
	Ok(())
}

/// External weights stay where they are: hard-link them next to the rewritten
/// graph, which refers to them by relative path (a copy if the cache is on
/// another filesystem). Not a symlink: ONNX Runtime resolves those and refuses
/// data whose real path lies outside the graph's directory.
fn link_external(src: &Path, dst: &Path, location: &str) -> Result<()> {
	let rel = Path::new(location);
	if rel.is_absolute() || rel.components().any(|c| matches!(c, Component::ParentDir | Component::Prefix(_))) {
		return Err(pb::bad("external data location escapes the model directory"));
	}
	if let Some(parent) = dst.parent() {
		std::fs::create_dir_all(parent)?;
	}
	let target = std::fs::canonicalize(src)?;
	if std::fs::hard_link(&target, dst).is_ok() {
		return Ok(());
	}
	std::fs::copy(&target, dst)?;
	Ok(())
}

/// Appends NodeProtos (and the small `axes` tensors they need) to a GraphProto.
struct Nodes<'a> {
	g: &'a mut Vec<u8>,
	axes_as_input: bool,
}

impl Nodes<'_> {
	fn node(&mut self, op: &str, inputs: &[&str], output: &str, int_attrs: &[(&str, i64)]) {
		self.node_ints(op, inputs, output, int_attrs, None);
	}

	fn node_ints(&mut self, op: &str, inputs: &[&str], output: &str, int_attrs: &[(&str, i64)], ints_attr: Option<(&str, &[i64])>) {
		let mut n = Vec::new();
		for i in inputs {
			pb::put_len(&mut n, 1, i.as_bytes());
		}
		pb::put_len(&mut n, 2, output.as_bytes());
		pb::put_len(&mut n, 3, format!("{output}_rsp_{op}").as_bytes());
		pb::put_len(&mut n, 4, op.as_bytes());
		for (name, v) in int_attrs {
			let mut a = Vec::new();
			pb::put_len(&mut a, 1, name.as_bytes());
			pb::put_uint(&mut a, 3, *v as u64);
			pb::put_uint(&mut a, 20, 2); // AttributeType INT
			pb::put_len(&mut n, 5, &a);
		}
		if let Some((name, values)) = ints_attr {
			let mut a = Vec::new();
			pb::put_len(&mut a, 1, name.as_bytes());
			for v in values {
				pb::put_uint(&mut a, 8, *v as u64);
			}
			pb::put_uint(&mut a, 20, 7); // AttributeType INTS
			pb::put_len(&mut n, 5, &a);
		}
		pb::put_len(self.g, 1, &n);
	}

	fn cast(&mut self, input: &str, output: &str, to: u64) {
		self.node("Cast", &[input], output, &[("to", to as i64)]);
	}

	/// Unsqueeze / ReduceSum with `axes` as an input (opset >= 13) or attribute.
	fn with_axes(&mut self, op: &str, input: &str, axes: &[i64], keepdims: Option<i64>, output: &str) {
		let keep: Vec<(&str, i64)> = keepdims.map(|k| ("keepdims", k)).into_iter().collect();
		if self.axes_as_input {
			let name = format!("{output}_rsp_axes");
			let payload: Vec<u8> = axes.iter().flat_map(|a| a.to_le_bytes()).collect();
			put_tensor(self.g, &name, &[axes.len() as i64], INT64, &payload);
			self.node(op, &[input, &name], output, &keep);
		} else {
			self.node_ints(op, &[input], output, &keep, Some(("axes", axes)));
		}
	}
}

fn put_scalar(g: &mut Vec<u8>, name: &str, data_type: u64, payload: &[u8]) {
	put_tensor(g, name, &[], data_type, payload);
}

fn put_tensor(g: &mut Vec<u8>, name: &str, dims: &[i64], data_type: u64, payload: &[u8]) {
	let mut t = Vec::new();
	for &d in dims {
		pb::put_uint(&mut t, 1, d as u64);
	}
	pb::put_uint(&mut t, 2, data_type);
	pb::put_len(&mut t, 8, name.as_bytes());
	pb::put_len(&mut t, 9, payload);
	pb::put_len(g, 5, &t);
}

#[cfg(test)]
#[path = "tests/graph_pooling_tests.rs"]
mod tests;
