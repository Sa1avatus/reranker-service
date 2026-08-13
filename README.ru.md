# Reranker Service

**Русский** | [English](README.md)

Reranker Service — независимый сервис cross-encoder ранжирования на FastAPI с административной
консолью. Он оценивает универсальные пары `query + document` с помощью зафиксированной модели
`BAAI/bge-reranker-v2-m3@953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`. Production path по
умолчанию использует ONNX Runtime CUDA с управляемым CPU fallback. Поддерживаются кеширование оценок в
Redis, ограниченный динамический batching, пакетные запросы, наблюдаемость через
Prometheus/OpenTelemetry и безопасная деградация кеша.

## Возможности

- защищённые API одиночного и пакетного ранжирования;
- стабильное ранжирование с сохранением исходного порядка при равных оценках;
- одна неизменяемая ревизия и один активный runtime модели в контейнере;
- registry backend-реализаций и типизированные pairwise-capabilities существующего CrossEncoder;
- проверка, прогрев, активация и откат модели-кандидата;
- Redis-кеш с ключами только на основе SHA-256 и деградацией без остановки инференса;
- административная React-консоль с playground, бенчмарками, метриками, runtime-настройками и
  технической историей запросов с раскрывающимися деталями (query, результаты ранжирования, документы);
- версионированное локальное хранение в браузере последнего query, упорядоченных документов с
  metadata и top-N одиночного Playground; очистка выполняется явно, credentials не сохраняются;
- CPU- и CUDA-варианты Docker-образа и воспроизводимый PowerShell-установщик.

API доступен по адресу `http://localhost:8200`, административная консоль —
`http://localhost:8400`.

## Архитектура

Схема следует компактному стилю компонентов и потоков данных из `job-searching-assistant`: API
владеет контрактами и оркестрацией, runtime — инференсом модели, а Redis остаётся необязательным
кешем и не влияет на readiness.

```mermaid
flowchart LR
    Client["API-клиент"] --> API["FastAPI API ранжирования"]
    Admin["Администратор"] --> Web["React-консоль / Nginx"]
    Web --> API
    API --> Guard["Аутентификация, валидация, rate limits"]
    Guard --> Service["RerankService"]
    Service <--> Redis[("Redis-кеш оценок")]
    Service --> Batcher["DynamicBatcher"]
    Batcher --> Runtime["Единственный ModelRuntime"]
    Runtime --> Model["Зафиксированный BGE CrossEncoder"]
    ModelCache[("Docker volume кеша модели")] --> Runtime
    API --> Metrics["Prometheus / OpenTelemetry"]
```

Web-контейнер только раздаёт консоль и проксирует HTTP. FastAPI отвечает за аутентификацию,
валидацию, лимиты, стабильные контракты ответов и технический аудит. `RerankService` координирует
поиск в хешированном кеше и ограниченный инференс. `DynamicBatcher` объединяет пары, не смешивая
идентичность запросов, а `ModelRuntime` владеет единственным активным CrossEncoder и executor.
Сбой Redis уменьшает эффективность кеша, но не нарушает инференс или readiness.

## Установка с нуля в Windows

Требования: Windows 10 или 11, PowerShell 5.1 или новее, Docker Desktop с Compose v2, минимум
8 ГБ RAM и около 12 ГБ свободного места для exporter, runtime и artifact модели.

```powershell
git clone https://github.com/Sa1avatus/reranker-service.git
Set-Location reranker-service
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\install.ps1
```

Установщик проверяет Docker, создаёт игнорируемый `.env` с независимыми случайными ключами API и
администратора, явно готовит immutable ONNX artifact через exporter, собирает GPU runtime,
запускает Redis/API/web и ждёт локальной загрузки и прогрева. Существующий `.env` не
перезаписывается. Для хоста без NVIDIA passthrough используйте `.\scripts\install.ps1 -Cpu`.
Первая установка может занять 10–30 минут в зависимости от сети и CPU.

После успешного readiness откройте `http://localhost:8400`. Читайте токен администратора только из
локального `.env`; не помещайте его в логи и не коммитьте.

```powershell
docker compose ps
docker compose logs -f reranker-api
docker compose down
# Данные сохраняются. Добавляйте -v только для намеренного удаления модели и данных Redis.
```

Повторный запуск `.\scripts\install.ps1` безопасен. Используйте `-SkipBuild`, если образ актуален,
или `-ReadyTimeoutMinutes 60` при медленной первой загрузке.

## Ручной быстрый запуск

```powershell
Copy-Item .env.example .env
# Замените оба секретных значения-заглушки до запуска сервиса.
docker compose --profile exporter build reranker-exporter
docker compose --profile exporter run --rm reranker-exporter `
  --model-id BAAI/bge-reranker-v2-m3 `
  --revision 953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e `
  --precision fp32 `
  --score-transform sigmoid
docker compose up -d --build
Invoke-RestMethod http://localhost:8200/health/ready
```

Для локальной разработки допустимы временные development-секреты. Не используйте их в deployment.

```powershell
$body = @{
  query = 'Опыт работы с Kubernetes'
  documents = @(
    @{ id = 'a'; text = 'Эксплуатация Kubernetes в production' }
    @{ id = 'b'; text = 'Администрирование SQL Server' }
  )
  top_n = 2
} | ConvertTo-Json -Depth 4

Invoke-RestMethod -Method Post -Uri http://localhost:8200/v1/rerank `
  -Headers @{ Authorization = 'Bearer <ваш-локальный-api-ключ>' } `
  -ContentType 'application/json' -Body $body
```

## Локальная разработка

Требуется Python 3.12. Unit-тесты используют детерминированный mock runtime и не загружают
production-модель.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[legacy,dev]"
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m mypy reranker_service
.\.venv\Scripts\python.exe -m pytest
```

Для консоли нужен Node.js 22. Из каталога `web/` выполните `npm install`, `npm test` и
`npm run build`.

## Эксплуатация и безопасность

- Один API worker выбран намеренно: несколько worker дублируют модель в памяти.
- CPU-инференсу обычно нужно 2–6 ГиБ в зависимости от длины последовательностей, batch и
  concurrency.
- CUDA дополнительно требует память под веса, активации и allocator. При OOM уменьшайте max length,
  batch size или concurrency.
- Входной текст, bearer credentials и нехешированные входы кеша не логируются и не сохраняются.
- `score` содержит raw logit модели. `normalized_score` возвращается только при явно заданной
  model-specific нормализации; оценки нельзя сравнивать между ревизиями.

См. [API.md](API.md), [ARCHITECTURE.md](ARCHITECTURE.md),
[OPERATIONS.md](OPERATIONS.md), [DEVELOPMENT.md](DEVELOPMENT.md) и
[SECURITY.md](SECURITY.md). Измеренные результаты сохранены в [BENCHMARKS.md](BENCHMARKS.md).

## ONNX artifact и GPU-контур

API-контейнер не скачивает и не экспортирует модели. Отдельный target `exporter` принимает только модель из allowlist, преобразует `main`, tag, branch или короткий SHA в полный неизменяемый 40-символьный commit SHA, запускает Optimum-валидацию и сохраняет `model.onnx`, `tokenizer.json` и versioned manifest с SHA-256 checksums. Параметр `--score-transform` обязателен, потому что нормализация score зависит от конкретной модели.

```powershell
docker compose --profile exporter run --rm reranker-exporter `
  --model-id BAAI/bge-reranker-v2-m3 `
  --revision 953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e `
  --precision fp32 `
  --score-transform sigmoid
```

GPU runtime запускается через `docker-compose.gpu.yml`, использует ONNX Runtime 1.26.0, CUDA 12.8 и cuDNN 9 и не содержит PyTorch/SentenceTransformers. Перед CUDA он отдельно проверяет artifact и короткий inference на CPU, поэтому повреждённая модель не маскируется fallback. В режиме `device=auto` ошибка инициализации CUDA переводит уже проверенную модель на CPU, если fallback разрешён; readiness остаётся успешным, но возвращает `degraded=true` и причину. Принудительный `device=cuda` завершается fail-closed. Для CPU используйте самостоятельный стек `docker compose -f docker-compose.cpu.yml up -d --build`.

```powershell
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
Invoke-RestMethod http://localhost:8200/health/ready
```

Базовый Compose использует проверенный ONNX GPU target с `device=auto`. Самостоятельный CPU-стек не запрашивает GPU и выбирает `CPUExecutionProvider`; legacy CrossEncoder остаётся отдельным target для parity и rollback. Admin UI и API показывают backend, requested/resolved revision, active/available providers, GPU name, fallback provider и degraded reason.

## Альтернативные модели

Jina v3 и Alibaba GTE запускаются только через отдельные opt-in Compose overrides с неизменяемыми
SHA модели, явным `RERANKER_TRUST_REMOTE_CODE=true` и точным allowlist:

```powershell
docker compose -f docker-compose.yml -f docker-compose.jina.yml up -d --build
docker compose -f docker-compose.yml -f docker-compose.alibaba.yml up -d --build
```

`jina_listwise` выполняет настоящий listwise inference для полного упорядоченного набора до 64
документов. Для него отключены pair-cache и batching разных запросов. Сырая оценка — cosine
similarity, normalized score переводит диапазон `[-1, 1]` в `[0, 1]`. Лицензия Jina v3 —
CC-BY-NC-4.0, поэтому перед использованием нужна отдельная проверка условий.

`alibaba_gte` использует изолированный Transformers 4.39.1 и фиксирует как SHA модели, так и SHA
вторичного репозитория `Alibaba-NLP/new-impl`; logit нормализуется sigmoid. Qwen3 Reranker, Ettin и
MiniLM проверены в отдельном `runtime-legacy`. Эти runtime не входят в ONNX production image по
умолчанию.
