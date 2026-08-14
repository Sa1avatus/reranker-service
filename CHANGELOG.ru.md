# История изменений

**Русский** | [English](CHANGELOG.md)

Все существенные изменения Reranker Service документируются в этом файле. Формат следует
[Keep a Changelog](https://keepachangelog.com/ru/1.1.0/), а для этапов разработки используется
семантическое версионирование.

В репозитории пока нет release tags. Версии ниже восстанавливают основные этапы по Git-истории;
метаданные Python-пакета, API и web синхронизированы на версии `0.4.1`.

## [Unreleased]

### Добавлено

- Добавлен дополнительный локальный multi-backend Compose-стек с отдельными контейнерами Jina,
  Alibaba и legacy CrossEncoder за единым Nginx API proxy.
- Добавлены статическое обнаружение backend через `GET /v1/backends` и маршрутизация по
  `X-Backend` для экспериментального multi-backend стека.
- Добавлены прямые локальные порты для диагностики: `8210` для Jina, `8211` для Alibaba и `8212`
  для legacy runtime.

### Изменено

- Выбор backend переведён на registry-контракт: UI использует серверный default, сохраняет только
  валидный явный выбор и не отправляет routing header при недоступном discovery.
- Сохранение текстов Playground по умолчанию выключено и доступно только как явный opt-in.
- Публикуемые Compose-порты привязаны к loopback, а web-команды стандартизированы на pnpm.
- Версии Python-пакета, API и web синхронизированы на `0.4.1`.
- Увеличены таймауты web proxy для долгих операций загрузки модели и ранжирования; в API теперь
  передаются стандартные заголовки клиента и исходного протокола.
- Web-пакет объявлен явным pnpm workspace, а install script `esbuild` добавлен в allowlist.
- Обновлена английская и русская onboarding-документация, добавлена эта история изменений.

### Исправлено

- Evaluation stack больше не загружает три модели в GPU одновременно: selector останавливает все
  model-контейнеры и запускает ровно один явно выбранный backend.
- Jina удалена как неявный UI fallback; неизвестные backend identifiers отклоняются предсказуемо.
- Web proxy больше не завершает корректные долгие запросы `/v1/*` по прежнему короткому таймауту.
- Placeholder-конфигурация разрешений pnpm заменена валидным workspace-описанием.

## [0.4.1] - 2026-08-13

### Добавлено

- В admin console добавлены раскрывающиеся детали запросов: query, упорядоченные документы,
  результаты, raw и normalized scores, rank, возвращённый текст и состояние cache hit.
- В таблицу и панель деталей добавлены timestamps запросов.
- Граница хранения истории запросов описана в API- и security-документации.

### Изменено

- In-memory preview ограничен 500 символами для query и 200 символами для каждого документа.
- Payload истории остаётся только в ограниченной process-local очереди и не записывается в logs,
  внешнюю telemetry или постоянное серверное хранилище.

### Исправлено

- Для старых записей без query, documents или results в консоли показывается понятный fallback.
- Панель деталей корректно обрабатывает частично заполненные исторические записи.

## [0.4.0] - 2026-08-11

### Добавлено

- Добавлен registry backend-реализаций с изолированными ONNX pairwise, legacy CrossEncoder,
  Alibaba GTE pairwise и Jina listwise.
- Добавлены versioned ONNX manifests с проверкой checksums и привязкой к model ID, immutable
  revision, backend и precision.
- Добавлен отдельный model exporter, который разрешает ревизии в полные commit SHA и никогда не
  запускается при старте API.
- Добавлены ONNX Runtime CUDA/CPU images и отдельные legacy, Jina и Alibaba runtime targets.
- Добавлены model-specific семантика raw score и опциональное поле `normalized_score`.
- Добавлены provider-aware readiness, проверка CUDA, управляемый CPU fallback и GPU diagnostics.
- Добавлено versioned browser-local хранение query, упорядоченных документов, metadata и top-N
  одиночного Playground.
- Добавлены импорт документов из JSON, CSV и plain text и детерминированное изменение порядка.
- Добавлены воспроизводимые benchmarks latency, throughput, ranking parity, hard negatives, размеров
  images и совместимости альтернативных моделей.

### Изменено

- `onnx_pairwise` с CUDA-first provider selection стал production backend по умолчанию.
- Remote-code backends оставлены opt-in и изолированы immutable ревизиями модели/кода, явным trust
  configuration и точными allowlists.
- Для Jina отключены per-pair cache и batching разных запросов, потому что listwise score зависит от
  полного набора кандидатов.

### Исправлено

- ONNX artifact сначала проверяется через CPU session, поэтому повреждённый graph больше не
  маскируется как ошибка инициализации CUDA.
- Ошибка проверки, прогрева или активации модели-кандидата не останавливает активную модель.
- `device=auto` деградирует только к отдельно проверенной CPU session; принудительный CUDA остаётся
  fail-closed.

### Безопасность

- API runtime больше не скачивает и не экспортирует model code или weights неявно.
- Добавлены path confinement, проверки symlink, schema/identity manifests и SHA-256 runtime
  artifacts.
- Для Jina и Alibaba требуется явный immutable allowlist remote code.

## [0.3.0] - 2026-08-09

### Добавлено

- Добавлены проверка, прогрев, активация, rollback и lifecycle reporting модели-кандидата.
- Добавлены проверки memory headroom и состояние restart-required, когда две модели нельзя безопасно
  держать в памяти одновременно.
- Добавлен воспроизводимый PowerShell installer для Windows и CPU-only installation.
- Добавлена синхронизированная английская и русская onboarding-документация с architecture diagrams.

### Исправлено

- Неудачная загрузка кандидата больше не заменяет и не выгружает активную модель.
- Недостаток GPU memory возвращается как явный lifecycle result без спекулятивного завершения
  активного runtime.

## [0.2.0] - 2026-08-08

### Добавлено

- Добавлен ограниченный cross-request dynamic micro-batching с лимитами пар, tokens, queue,
  timeout и concurrency.
- Добавлены runtime/cache admin endpoints, запуск benchmarks, агрегированные metrics, health views,
  request history и operational dashboard panels.
- Добавлены authenticated batch reranking и расширенные административные workflows.
- Добавлены импорт документов и стабильное изменение их порядка в Playground.
- Добавлены opt-in Redis integration tests для TTL, hashed keys, ACL failures, degraded inference и
  восстановления.

### Исправлено

- При объединении нескольких запросов в один inference batch сохраняются request identity и порядок
  результатов.
- Cancellation или timeout одного запроса не повреждает сопоставление результатов другого запроса.
- Сбой Redis снижает эффективность кеша, но не ломает readiness и inference; после восстановления
  Redis кеш снова используется автоматически.

## [0.1.0] - 2026-08-08

### Добавлено

- Добавлен первый production-oriented FastAPI сервис с authenticated контрактами одиночного и
  пакетного `query + document` reranking.
- Добавлена стабильная сортировка с сохранением исходного порядка при одинаковых scores.
- Добавлены одна immutable revision модели и один model runtime на контейнер.
- Добавлен Redis score cache с ключами только на основе SHA-256.
- Добавлены request validation, ограничение входных данных, структурированные ошибки,
  readiness/liveness, Prometheus metrics, OpenTelemetry и React admin console.
- Добавлены Docker, основы CPU/GPU runtime, unit tests, Ruff, mypy и coverage gates.

### Безопасность

- Bearer credentials, raw cache inputs и обычный текст запросов исключены из application logs.
- Добавлены constant-time bearer comparison, non-root containers, CSP/security headers и
  обобщённые ответы на internal errors.
