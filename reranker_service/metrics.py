from prometheus_client import Counter, Gauge, Histogram

HTTP_REQUESTS = Counter("reranker_http_requests_total", "HTTP requests", ["endpoint", "status"])
HTTP_DURATION = Histogram("reranker_http_request_duration_seconds", "HTTP latency", ["endpoint"])
IN_PROGRESS = Gauge("reranker_requests_in_progress", "Requests in progress")
DOCUMENTS_RECEIVED = Counter("reranker_documents_received_total", "Documents received")
DOCUMENTS_SCORED = Counter("reranker_documents_scored_total", "Documents scored")
INFERENCE_DURATION = Histogram(
    "reranker_inference_duration_seconds", "Inference latency", ["model", "device"]
)
BATCH_SIZE = Histogram("reranker_batch_size", "Inference batch size")
QUEUE_WAIT = Histogram("reranker_queue_wait_seconds", "Queue wait")
CACHE_HITS = Counter("reranker_cache_hits_total", "Cache hits")
CACHE_MISSES = Counter("reranker_cache_misses_total", "Cache misses")
CACHE_ERRORS = Counter("reranker_cache_errors_total", "Cache errors")
MODEL_LOAD = Histogram("reranker_model_load_seconds", "Model load latency", ["model", "device"])
MODEL_READY = Gauge("reranker_model_ready", "Model readiness", ["model", "device"])
TRUNCATIONS = Counter("reranker_truncations_total", "Inputs truncated")
RATE_LIMITED = Counter("reranker_rate_limited_total", "Rate limited requests")
TIMEOUTS = Counter("reranker_timeouts_total", "Inference timeouts")
ERRORS = Counter("reranker_errors_total", "Service errors", ["status"])
PROVIDER_ACTIVE = Gauge(
    "reranker_provider_active", "Active inference execution provider", ["backend", "provider"]
)
PROVIDER_FALLBACKS = Counter(
    "reranker_provider_fallbacks_total",
    "Controlled execution provider fallbacks",
    ["backend", "provider", "reason"],
)
CUDA_AVAILABLE = Gauge("reranker_cuda_available", "Whether CUDA provider is available")
CUDA_INITIALIZATION_FAILURES = Counter(
    "reranker_cuda_initialization_failures_total", "CUDA session initialization failures"
)
CUDA_INFERENCE_FAILURES = Counter(
    "reranker_cuda_inference_failures_total", "CUDA inference failures"
)
CUDA_OOM = Counter("reranker_cuda_oom_total", "CUDA out-of-memory failures")
GPU_MEMORY_USED = Gauge("reranker_gpu_memory_used_bytes", "GPU memory currently used")
GPU_MEMORY_FREE = Gauge("reranker_gpu_memory_free_bytes", "GPU memory currently free")
