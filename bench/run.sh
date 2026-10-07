#!/usr/bin/env bash
# Load-test a running rs-infer server with fixed scenarios; one line per scenario.
#
#   bench/run.sh [URL] [SCENARIO...]
#
# URL defaults to http://127.0.0.1:8080. Scenarios (default: s1 s2 h1 r1):
#   s1  1 short query per request, 64 concurrent clients      (search traffic)
#   s2  32 passages of ~90 tokens per request, 8 clients      (ingestion)
#   h1  64 texts of 3-250 words per request, 4 clients        (mixed lengths)
#   r1  rerank, 1 query + 1 document per request, 64 clients  (needs a rerank model)
# Requests carry no `model`, so the server's default model of each kind answers.
# DURATION (default 20s) sets the measured time per scenario, after a 3s warmup.
# Needs `oha` (cargo install oha) and python3.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
url=${1:-http://127.0.0.1:8080}
[ $# -gt 0 ] && shift
scenarios=${*:-s1 s2 h1 r1}
duration=${DURATION:-20s}

run() { # name path body concurrency texts-per-request
	local out
	out=$(mktemp)
	oha -z 3s -c "$4" -m POST -H 'content-type: application/json' -D "$here/bodies/$3" --no-tui "$url$2" >/dev/null 2>&1 || true
	oha -z "$duration" -c "$4" -m POST -H 'content-type: application/json' -D "$here/bodies/$3" --no-tui --output-format json "$url$2" >"$out"
	python3 "$here/summarize.py" "$1" "$5" "$out"
	rm -f "$out"
}

for s in $scenarios; do
	case $s in
		s1) run s1-query /v1/embeddings s1.json 64 1 ;;
		s2) run s2-ingest /v1/embeddings s2.json 8 32 ;;
		h1) run h1-mixed /v1/embeddings het.json 4 64 ;;
		r1) run r1-rerank /v1/rerank r1.json 64 1 ;;
		*) echo "unknown scenario '$s' (s1 s2 h1 r1)" >&2; exit 1 ;;
	esac
done
