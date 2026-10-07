//! Unit tests for [`zeroshot`](super).

use super::*;
use crate::pipeline::Fwd;

#[test]
fn entailment_minus_contradiction_logits() {
	let d = [0.1f32, 2.0, 0.5, 1.5, 0.2, 0.3];
	let fwd = Fwd { shape: vec![2, 3], data: &d };
	let s = entailment_logits(&fwd, 1, 0).unwrap();
	assert!((s[0] - 1.9f64).abs() < 1e-6);
	assert!((s[1] + 1.3f64).abs() < 1e-6);
}

#[test]
fn softmax_sums_to_one() {
	let v = softmax_f64(&[1.0, 2.0, 3.0]);
	assert!((v.iter().sum::<f64>() - 1.0).abs() < 1e-9);
	assert!(v[2] > v[1] && v[1] > v[0]);
}

#[test]
fn inverts_yesno_questions() {
	assert_eq!(invert_question("Is this email important?").as_deref(), Some("This email is important."));
	assert_eq!(invert_question("Does the package arrive today?").as_deref(), Some("The package does arrive today."));
	assert_eq!(invert_question("Is this urgent?").as_deref(), Some("This is urgent."));
	assert_eq!(invert_question("What is the weather?"), None); // not a yes/no question
	assert_eq!(invert_question("Is important?"), None); // too short
}

#[test]
fn inverts_questions_with_non_ascii_subject() {
	assert_eq!(invert_question("Is élan vital important?").as_deref(), Some("Élan vital is important."));
	assert_eq!(invert_question("Is Émile here?").as_deref(), Some("Émile is here."));
}
