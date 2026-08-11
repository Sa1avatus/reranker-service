# Verified benchmarks

These measurements were captured on 2026-08-11 on the production host's NVIDIA GeForce RTX 3060
through Docker Desktop/WSL2. The model was `BAAI/bge-reranker-v2-m3` at immutable revision
`953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`. ONNX used opset 18, FP32, ONNX Runtime 1.26.0,
CUDA 12.8 and cuDNN 9. Results include tokenizer and inference time but exclude model load.
Three measured repetitions followed one warm-up, so p95/p99 below are directional rather than a
production SLO study.

## Short-input latency and throughput

Each document contained approximately 16 words.

| Runtime | Documents | p50 ms | observed p95 ms | pairs/s |
| --- | ---: | ---: | ---: | ---: |
| ONNX CUDA FP32 | 1 | 15.49 | 20.98 | 62.18 |
| ONNX CUDA FP32 | 10 | 52.39 | 55.25 | 189.74 |
| ONNX CUDA FP32 | 50 | 171.53 | 172.44 | 291.57 |
| ONNX CUDA FP32 | 100 | 336.32 | 337.81 | 297.00 |
| ONNX CPU FP32 | 1 | 123.52 | 124.23 | 8.70 |
| ONNX CPU FP32 | 10 | 746.06 | 964.32 | 12.54 |
| ONNX CPU FP32 | 50 | 3064.98 | 3189.10 | 16.27 |
| ONNX CPU FP32 | 100 | 5893.19 | 7042.89 | 16.01 |
| legacy PyTorch CUDA FP16 | 1 | 59.82 | 68.50 | 17.58 |
| legacy PyTorch CUDA FP16 | 10 | 52.91 | 60.00 | 205.02 |
| legacy PyTorch CUDA FP16 | 50 | 140.71 | 144.87 | 376.31 |
| legacy PyTorch CUDA FP16 | 100 | 199.99 | 218.71 | 489.88 |

ONNX CUDA is substantially faster than CPU fallback and has lower single-pair latency than the
legacy path. The current legacy FP16 implementation remains faster for large short-text batches;
the ONNX graph emitted a memcpy-node optimization warning, so this is a documented optimization
opportunity rather than a claim that ONNX wins every workload.

ONNX CUDA p50 for 100 documents was 818.44 ms at roughly 64 words and 3195.42 ms at roughly 256
words. A full CPU matrix including 50/100 documents at 256 words exceeded the 15-minute benchmark
limit and was stopped; it is not reported as a successful result.

## Ranking parity and hard-negative regression

The committed `tests/data/job_relevance_cases.json` contains six relevant engineering roles and six
hard negatives, including Sous Chef. ONNX CPU and CUDA produced identical rankings. Against legacy
CUDA, ONNX CUDA produced Spearman 0.9930, Kendall tau 0.9697, top-1/top-3 agreement 1.0, and top-10
overlap 0.9. All three runtimes ranked the six relevant roles first; Sous Chef ranked 10th or 11th
of 12 with a normalized ONNX score near 0.0000165.

For a two-pair numeric smoke case, PyTorch FP32 normalized scores were
`0.9995762706/0.0000161589`, ONNX CPU `0.9995763251/0.0000161589`, and ONNX CUDA
`0.9995766685/0.0000161591`. Legacy CUDA FP16 raw logits were `7.765625/-11.03125` versus the FP32
baseline `7.766109/-11.033023`.

## Runtime image sizes

Docker's inspected image sizes were 4,094,913,702 bytes for the prior production legacy image,
3,210,547,857 bytes for the final ONNX GPU runtime, 117,046,503 bytes for the final ONNX CPU runtime,
and 408,004,221 bytes for the final separate exporter. The ONNX GPU runtime is approximately 21.6% smaller
than the prior production image and contains neither `torch` nor `sentence-transformers`.

## Model compatibility smoke matrix

The following CUDA FP16 smokes ran on the same RTX 3060 host. Each test used an immutable model
revision and compared one relevant software-engineering document with an unrelated hard negative.
These values prove load and ordering compatibility only; they are not cross-model quality scores.

| Backend/runtime | Immutable model revision | Raw scores (relevant / negative) | Result |
| --- | --- | ---: | --- |
| `jina_listwise` | `jinaai/jina-reranker-v3@d7d7e73b6ea138ced340b83865931b5dfb6c97aa` | `0.4382 / -0.1653` (four-document run) | relevant first |
| `alibaba_gte` | `Alibaba-NLP/gte-multilingual-reranker-base@8215cf04918ba6f7b6a62bb44238ce2953d8831c` | `1.0107 / -1.7627` | relevant first |
| `legacy_cross_encoder` | `Qwen/Qwen3-Reranker-0.6B@e61197ed45024b0ed8a2d74b80b4d909f1255473` | `8.0469 / -11.2031` | relevant first |
| `legacy_cross_encoder` | `cross-encoder/ettin-reranker-17m-v1@9e4aa35321a6dd1a43ca313f500c4b4f7cfb5cc6` | `11.7500 / 2.8281` | relevant first |
| `legacy_cross_encoder` | `cross-encoder/ms-marco-MiniLM-L6-v2@233902d25c440f23af6f7d6e94d2946bac0bee0a` | `9.0469 / -11.3438` | relevant first |

Alibaba also pinned `Alibaba-NLP/new-impl@40ced75c3017eb27626c9d4ea981bde21a2662f4`.
Its official custom architecture failed under the shared modern Transformers stack, then passed in
the isolated Transformers 4.39.1 image declared by the checkpoint. Ettin required the current
Sentence Transformers 5.4.1 / Transformers 5.7.0 compatibility target. Jina used its native
listwise API; per-pair caching and dynamic batching were not involved.
