//! Unit tests for [`rerank`](super).

use super::*;
use crate::{config::Scoring, pipeline::Fwd};

#[test]
fn logit_and_sigmoid_single_column() {
	let d: Vec<f32> = vec![0.0, 2.0];
	let fwd = Fwd { shape: vec![2, 1], data: &d };
	let s = apply_scoring(&fwd, Scoring::Logit, None, None, &[vec![1], vec![1]]).unwrap();
	assert_eq!(s[0], 0.0);
	let s = apply_scoring(&fwd, Scoring::Sigmoid, None, None, &[vec![1], vec![1]]).unwrap();
	assert!((s[0] - 0.5).abs() < 1e-9);
	assert!(s[1] > s[0]);
}

#[test]
fn auto_single_column_is_normalized_0_1() {
	let d: Vec<f32> = vec![-11.0, 0.0, 8.5];
	let fwd = Fwd { shape: vec![3, 1], data: &d };
	let s = apply_scoring(&fwd, Scoring::Auto, None, None, &[vec![1], vec![1], vec![1]]).unwrap();
	assert!(s.iter().all(|v| *v > 0.0 && *v < 1.0), "scores out of range: {s:?}");
	assert!(s[2] > s[1] && s[1] > s[0]); // ranking preserved
	assert!((s[1] - 0.5).abs() < 1e-9);
}

#[test]
fn softmax_two_column() {
	let d: Vec<f32> = vec![0.0, 0.0];
	let fwd = Fwd { shape: vec![1, 2], data: &d };
	let s = apply_scoring(&fwd, Scoring::Auto, None, None, &[vec![1]]).unwrap();
	assert!((s[0] - 0.5).abs() < 1e-9);
}

#[test]
fn sigmoid_two_column_is_sigmoid_of_logit_diff() {
	let d: Vec<f32> = vec![0.0, 0.0, /* row 1 */ 0.0, 3.0];
	let fwd = Fwd { shape: vec![2, 2], data: &d };
	let s = apply_scoring(&fwd, Scoring::Sigmoid, None, None, &[vec![1], vec![1]]).unwrap();
	assert!((s[0] - 0.5).abs() < 1e-9, "got {}", s[0]);
	let expect = 1.0 / (1.0 + (-3.0f64).exp());
	assert!((s[1] - expect).abs() < 1e-9, "got {}", s[1]);
}

#[test]
fn yes_no_uses_last_real_token() {
	// [B=1, S=2, V=3]: token ids: yes=1, no=2; both positions real -> score at last
	let d: Vec<f32> = vec![0.0, 0.0, 10.0, /* pos 1 */ 0.0, 5.0, 0.0];
	let fwd = Fwd { shape: vec![1, 2, 3], data: &d };
	let attn = vec![vec![1i64, 1]];
	let s = apply_scoring(&fwd, Scoring::YesNo, Some(1), Some(2), &attn).unwrap();
	let expect = 1.0 / (1.0 + (-5.0f64).exp());
	assert!((s[0] - expect).abs() < 1e-6, "got {}", s[0]);
}
