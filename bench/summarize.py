"""One summary line from an `oha --output-format json` report (used by run.sh)."""

import json
import sys

name, texts_per_request, path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
report = json.load(open(path))
summary, latency = report["summary"], report.get("latencyPercentiles") or {}
codes = report.get("statusCodeDistribution") or {}
ok = codes.get("200", 0)
total = sum(codes.values()) + sum((report.get("errorDistribution") or {}).values())
# Requests still in flight when the timer stops count as errors; scale them out.
rps = summary["requestsPerSec"] * (ok / total if total else 0)
ms = lambda key: (latency.get(key) or 0) * 1e3  # noqa: E731
print(
    f"{name:<17} {rps:8.1f} req/s {rps * texts_per_request:9.1f} texts/s"
    f"   p50 {ms('p50'):7.1f} ms   p99 {ms('p99'):7.1f} ms   ok {ok}/{total}"
)
