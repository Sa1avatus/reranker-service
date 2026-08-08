# Benchmarks

Start the service, then run `make benchmark`. The tool exercises multilingual English/Russian cases and reports p50/p95/p99, mean latency, and pairs/s. Use cold, warm, cache-enabled and cache-disabled runs separately; record device, batch size, max length, model revision, CPU RAM and CUDA memory alongside results. No benchmark result is committed because it would be hardware-dependent.

Admin benchmark records provide scheduling/baseline metadata. The bundled CLI performs the actual HTTP benchmark. Exclusive mode requires explicit `EXCLUSIVE` confirmation and should be isolated from production traffic.

