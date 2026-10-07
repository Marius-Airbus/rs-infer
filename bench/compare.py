"""Compare two servers' embeddings of the same texts (cosine per text).

Checks that a faster setup (GPU, fp16, int8, pooling in the graph) returns the
same vectors as a reference, e.g. a CPU server with `dtype: fp32`:

    python3 bench/compare.py http://gpu-host:8080 http://127.0.0.1:8081 [bodies/s2.json]

Uses the default embedding model of each server. Standard library only.
"""

import json
import math
import sys
import urllib.request
from pathlib import Path


def embed(url: str, texts: list[str]) -> list[list[float]]:
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/embeddings",
        data=json.dumps({"input": texts}).encode(),
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return [d["embedding"] for d in json.loads(resp.read())["data"]]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    body = Path(sys.argv[3]) if len(sys.argv) > 3 else Path(__file__).parent / "bodies" / "s2.json"
    texts = json.loads(body.read_text(encoding="utf-8"))["input"]
    texts = [texts] if isinstance(texts, str) else texts
    a, b = embed(sys.argv[1], texts), embed(sys.argv[2], texts)
    if len(a[0]) != len(b[0]):
        sys.exit(f"different dimensions: {len(a[0])} vs {len(b[0])} (not the same model?)")
    cos = [cosine(x, y) for x, y in zip(a, b)]
    print(f"{len(cos)} texts: cosine min {min(cos):.5f}, mean {sum(cos) / len(cos):.5f}")


if __name__ == "__main__":
    main()
