# Benchmarks

Start the service, then run `make benchmark`. The tool exercises multilingual English/Russian cases and reports p50/p95/p99, mean latency, and pairs/s. Use cold, warm, cache-enabled and cache-disabled runs separately; record device, batch size, max length, model revision, CPU RAM and CUDA memory alongside results. No benchmark result is committed because it would be hardware-dependent.

The admin API and web console execute a real background benchmark against the built-in English/Russian dataset. Runs include warm-up, p50/p95/p99, mean latency, pairs/documents per second, and process memory before/after. `low_priority` yields between cases; exclusive mode requires explicit `EXCLUSIVE` confirmation and should be isolated from production traffic. Only parameters and numeric results are retained.
