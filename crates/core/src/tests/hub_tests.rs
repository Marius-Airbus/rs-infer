//! Unit tests for [`hub`](super).

use super::*;

fn s(list: &[&str]) -> Vec<String> {
	list.iter().map(|x| x.to_string()).collect()
}

#[test]
fn prefers_root_fp32() {
	let sib = s(&["model.onnx", "onnx/model.onnx", "tokenizer.json", "config.json"]);
	assert_eq!(select_remote(&sib, None, MODEL_CANDIDATES).unwrap(), "model.onnx");
}

#[test]
fn falls_back_to_onnx_subfolder() {
	let sib = s(&["onnx/model.onnx", "onnx/model.onnx_data", "tokenizer.json"]);
	assert_eq!(select_remote(&sib, None, MODEL_CANDIDATES).unwrap(), "onnx/model.onnx");
}

#[test]
fn quantized_only_repo_picks_fp16_then_quantized() {
	// e.g. onnx-community repos that only ship quantized graphs
	let sib = s(&["onnx/model_q4.onnx", "onnx/model_quantized.onnx"]);
	assert_eq!(select_remote(&sib, None, MODEL_CANDIDATES).unwrap(), "onnx/model_quantized.onnx");
}

#[test]
fn explicit_file_wins() {
	let sib = s(&["onnx/model_q4.onnx", "onnx/model_int8.onnx"]);
	let cfg = crate::config::ModelConfig::default_for_test(Some("onnx/model_int8.onnx"));
	assert_eq!(select_remote(&sib, cfg.subfolder.as_deref(), &candidates_for(&cfg, MODEL_CANDIDATES)).unwrap(), "onnx/model_int8.onnx");
}

#[test]
fn download_requires_hf() {
	let cfg = crate::config::ModelConfig::default_for_test(None);
	let err = download_to(&cfg, Path::new("unused"), None).unwrap_err();
	assert!(matches!(err, Error::Config(_)));
}

#[test]
fn honors_hf_endpoint_and_token_env() {
	use std::{io::{Read, Write}, time::{Duration, Instant}};

	// A local stand-in for the Hub: records the first request and answers 404.
	let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
	let addr = listener.local_addr().unwrap();
	listener.set_nonblocking(true).unwrap();
	let hub = std::thread::spawn(move || {
		let deadline = Instant::now() + Duration::from_secs(10);
		let mut conn = loop {
			match listener.accept() {
				Ok((conn, _)) => break conn,
				Err(_) if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(10)),
				Err(_) => return String::new(), // the request never came here
			}
		};
		conn.set_nonblocking(false).unwrap();
		conn.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
		let mut head = Vec::new();
		let mut buf = [0u8; 1024];
		while !head.windows(4).any(|w| w == b"\r\n\r\n") {
			match conn.read(&mut buf) {
				Ok(0) | Err(_) => break,
				Ok(n) => head.extend_from_slice(&buf[..n]),
			}
		}
		let _ = conn.write_all(b"HTTP/1.1 404 Not Found\r\ncontent-length: 0\r\nconnection: close\r\n\r\n");
		String::from_utf8_lossy(&head).into_owned()
	});

	// Only this test reads or writes these variables.
	std::env::set_var("HF_ENDPOINT", format!("http://{addr}"));
	std::env::set_var("HF_TOKEN", "hf_test_token");
	let cfg = crate::config::ModelConfig { hf: Some("acme/missing-model".into()), ..crate::config::ModelConfig::default_for_test(None) };
	let err = resolve(&cfg, None).unwrap_err();
	std::env::remove_var("HF_ENDPOINT");
	std::env::remove_var("HF_TOKEN");

	let request = hub.join().unwrap();
	assert!(matches!(err, Error::Hub(_)), "{err}");
	assert!(request.starts_with("GET /api/models/acme/missing-model"), "request did not reach HF_ENDPOINT: {request:?}");
	assert!(request.to_ascii_lowercase().contains("authorization: bearer hf_test_token"), "HF_TOKEN not sent: {request:?}");
}

#[test]
fn int8_prefers_the_fp32_graph_it_can_quantize() {
	let cfg = crate::config::ModelConfig { dtype: crate::config::Dtype::Int8, ..crate::config::ModelConfig::default_for_test(None) };
	let both = s(&["onnx/model.onnx", "onnx/model_int8.onnx"]);
	assert_eq!(select_remote(&both, None, &candidates_for(&cfg, MODEL_CANDIDATES)).unwrap(), "onnx/model.onnx");
	// No fp32 graph: the publisher's int8 file.
	let only_int8 = s(&["onnx/model_int8.onnx", "onnx/model_q4.onnx"]);
	assert_eq!(select_remote(&only_int8, None, &candidates_for(&cfg, MODEL_CANDIDATES)).unwrap(), "onnx/model_int8.onnx");
}
