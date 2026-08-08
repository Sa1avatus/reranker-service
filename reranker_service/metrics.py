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
