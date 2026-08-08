import argparse
import json
import statistics
import time
from pathlib import Path

import httpx


def percentile(values: list[float], p: float) -> float:
    return sorted(values)[min(round((len(values) - 1) * p), len(values) - 1)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8200")
    parser.add_argument("--key", default="local-reranker-development-key")
    parser.add_argument("--repetitions", type=int, default=10)
    args = parser.parse_args()
    data = json.loads(Path(__file__).with_name("dataset.json").read_text(encoding="utf-8"))
    latencies = []
    pairs = 0
    with httpx.Client(timeout=30, headers={"Authorization": f"Bearer {args.key}"}) as client:
        for _ in range(args.repetitions):
            for case in data:
                body = {
                    "query": case["query"],
                    "documents": [
                        {"id": str(i), "text": x} for i, x in enumerate(case["documents"])
                    ],
                }
                started = time.perf_counter()
                response = client.post(args.url + "/v1/rerank", json=body)
                response.raise_for_status()
                latencies.append((time.perf_counter() - started) * 1000)
                pairs += len(case["documents"])
    seconds = sum(latencies) / 1000
    print(
        json.dumps(
            {
                "requests": len(latencies),
                "p50_ms": percentile(latencies, 0.5),
                "p95_ms": percentile(latencies, 0.95),
                "p99_ms": percentile(latencies, 0.99),
                "mean_ms": statistics.mean(latencies),
                "pairs_per_second": pairs / seconds,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
