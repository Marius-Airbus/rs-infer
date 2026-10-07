//! Unit tests for the shared pipeline helpers in [`pipeline`](super).

use super::*;

#[test]
fn direct_path_runs_shortest_first_and_restores_order() {
	let lens = [5, 1, 9, 3, 7];
	let rows: Vec<Row> = lens.iter().enumerate().map(|(i, &n)| Row { ids: vec![i as i64; n], type_ids: vec![0; n] }).collect();
	let mut batches = Vec::new();
	let out = run_sorted(rows, 2, 1000, |enc| {
		batches.push((enc.batch, enc.seq));
		Ok(enc.input_ids.iter().map(|r| RowOut::Score(r[0] as f64)).collect())
	})
	.unwrap();
	// Lengths 1,3 | 5,7 | 9: each batch pads only to its own longest row.
	assert_eq!(batches, [(2, 3), (2, 7), (1, 9)]);
	let ids: Vec<usize> = out.into_iter().map(|o| o.into_score().unwrap() as usize).collect();
	assert_eq!(ids, [0, 1, 2, 3, 4]);
}

#[test]
fn direct_path_respects_the_padded_token_budget() {
	let rows: Vec<Row> = [10, 10, 10].iter().map(|&n| Row { ids: vec![1; n], type_ids: vec![0; n] }).collect();
	let mut sizes = Vec::new();
	run_sorted(rows, 32, 20, |enc| {
		sizes.push(enc.batch);
		Ok(vec![RowOut::Score(0.0); enc.batch])
	})
	.unwrap();
	assert_eq!(sizes, [2, 1]);
}
