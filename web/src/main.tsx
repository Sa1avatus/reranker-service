import React, { useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import {
  QueryClient,
  QueryClientProvider,
  useMutation,
  useQuery,
  useQueryClient,
} from '@tanstack/react-query';

import './style.css';

const sections = [
  'Dashboard', 'Rerank Playground', 'Batch Playground', 'Models', 'Runtime', 'Cache',
  'Benchmarks', 'Requests', 'System Health', 'Settings', 'Audit Log',
] as const;
type Section = typeof sections[number];

const paths: Partial<Record<Section, string>> = {
  Dashboard: 'dashboard', Models: 'models', Benchmarks: 'benchmarks', Requests: 'requests',
  'System Health': 'system/health', Settings: 'runtime', 'Audit Log': 'audit-log',
};

export async function api<T>(url: string, token: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(`/v1/${url}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${token}`,
      ...options.headers,
    },
  });

  let payload: unknown = null;

  if (typeof response.text === 'function') {
    // Nginx/proxy errors may return HTML instead of JSON. Reading text first lets
    // the UI terminate the request cleanly instead of throwing "Unexpected token <".
    const raw = await response.text();
    if (raw) {
      try {
        payload = JSON.parse(raw);
      } catch {
        const compact = raw.replace(/\s+/g, ' ').trim().slice(0, 300);
        if (!response.ok) {
          throw new Error(
            `HTTP ${response.status} ${response.statusText}${compact ? `: ${compact}` : ''}`,
          );
        }
        throw new Error(`Invalid JSON response from /v1/${url}`);
      }
    }
  } else {
    // Keep compatibility with lightweight fetch implementations used by
    // embedders and tests, which may expose json() but not text().
    payload = await response.json();
  }

  if (!response.ok) {
    const errorPayload = payload as {
      error?: { message?: string };
      detail?: string;
      message?: string;
    } | null;
    throw new Error(
      errorPayload?.error?.message
      || errorPayload?.detail
      || errorPayload?.message
      || `HTTP ${response.status} ${response.statusText}`,
    );
  }

  return (payload ?? {}) as T;
}

export function Login({ onLogin }: { onLogin: (token: string) => void }) {
  const [token, setToken] = useState('');
  return <main className="login"><form onSubmit={(event) => {
    event.preventDefault();
    onLogin(token);
  }}><h1>Reranker Console</h1><p>Operational control plane</p><label>Admin token
    <input aria-label="Admin token" type="password" value={token}
      onChange={(event) => setToken(event.target.value)} autoFocus required />
  </label><button>Sign in</button></form></main>;
}

type PlaygroundDocument = { id: string; text: string; metadata: Record<string, unknown> };

export const PLAYGROUND_STORAGE_KEY = 'reranker.playground.v1';
const PLAYGROUND_STORAGE_VERSION = 1;
const PLAYGROUND_SAVE_DELAY_MS = 250;

type PlaygroundState = {
  version: 1;
  query: string;
  documents: PlaygroundDocument[];
  top_n: number;
};

const defaultPlaygroundState = (): PlaygroundState => ({
  version: PLAYGROUND_STORAGE_VERSION,
  query: "Describe the candidate's Kubernetes experience",
  documents: [
    { id: 'kubernetes', text: 'Basic conceptual knowledge of Kubernetes.', metadata: {} },
    { id: 'docker', text: 'Production experience with Docker Compose.', metadata: {} },
  ],
  top_n: 10,
});

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

export function loadPlaygroundState(): PlaygroundState {
  try {
    const raw = localStorage.getItem(PLAYGROUND_STORAGE_KEY);
    if (!raw) return defaultPlaygroundState();
    const value: unknown = JSON.parse(raw);
    if (!isRecord(value) || value.version !== PLAYGROUND_STORAGE_VERSION ||
        typeof value.query !== 'string' || !Array.isArray(value.documents) ||
        typeof value.top_n !== 'number' || !Number.isInteger(value.top_n) ||
        value.top_n < 1 || value.top_n > 100) {
      return defaultPlaygroundState();
    }
    const documents = value.documents.map((document): PlaygroundDocument | null => {
      if (!isRecord(document) || typeof document.id !== 'string' ||
          typeof document.text !== 'string' || !isRecord(document.metadata)) return null;
      return { id: document.id, text: document.text, metadata: document.metadata };
    });
    if (documents.some((document) => document === null) || documents.length > 100) {
      return defaultPlaygroundState();
    }
    return {
      version: PLAYGROUND_STORAGE_VERSION,
      query: value.query,
      documents: documents as PlaygroundDocument[],
      top_n: Number(value.top_n),
    };
  } catch {
    return defaultPlaygroundState();
  }
}

export function savePlaygroundState(state: Omit<PlaygroundState, 'version'>): void {
  try {
    if (!state.query && state.documents.length === 0 && state.top_n === 10) {
      localStorage.removeItem(PLAYGROUND_STORAGE_KEY);
      return;
    }
    const persisted: PlaygroundState = {
      version: PLAYGROUND_STORAGE_VERSION,
      query: state.query,
      documents: state.documents.map((document) => ({
        id: document.id,
        text: document.text,
        metadata: document.metadata,
      })),
      top_n: state.top_n,
    };
    localStorage.setItem(PLAYGROUND_STORAGE_KEY, JSON.stringify(persisted));
  } catch {
    // Storage can be unavailable or full; the Playground must remain usable.
  }
}

function csvRows(content: string): string[][] {
  const rows: string[][] = []; let row: string[] = []; let field = ''; let quoted = false;
  for (let index = 0; index < content.length; index += 1) {
    const character = content[index];
    if (character === '"' && quoted && content[index + 1] === '"') { field += '"'; index += 1; }
    else if (character === '"') quoted = !quoted;
    else if (character === ',' && !quoted) { row.push(field); field = ''; }
    else if ((character === '\n' || character === '\r') && !quoted) {
      if (character === '\r' && content[index + 1] === '\n') index += 1;
      row.push(field); if (row.some(Boolean)) rows.push(row); row = []; field = '';
    } else field += character;
  }
  row.push(field); if (row.some(Boolean)) rows.push(row);
  return rows;
}

export function parseDocuments(filename: string, content: string): PlaygroundDocument[] {
  const extension = filename.toLowerCase().split('.').pop();
  let documents: PlaygroundDocument[];
  if (extension === 'json') {
    const parsed: unknown = JSON.parse(content);
    const values = Array.isArray(parsed) ? parsed : (parsed as { documents?: unknown }).documents;
    if (!Array.isArray(values)) throw new Error('JSON must contain a document array');
    documents = values.map((value, index) => {
      if (typeof value === 'string') return { id: `doc-${index + 1}`, text: value, metadata: {} };
      if (!value || typeof value !== 'object' || typeof (value as { text?: unknown }).text !== 'string') {
        throw new Error(`Document ${index + 1} has no text`);
      }
      const object = value as { id?: unknown; text: string; metadata?: unknown };
      return { id: typeof object.id === 'string' ? object.id : `doc-${index + 1}`, text: object.text,
        metadata: object.metadata && typeof object.metadata === 'object' ?
          object.metadata as Record<string, unknown> : {} };
    });
  } else if (extension === 'csv') {
    const rows = csvRows(content); const headers = rows.shift()?.map((header) => header.trim().toLowerCase());
    const textIndex = headers?.indexOf('text') ?? -1;
    if (textIndex < 0) throw new Error('CSV requires a text column');
    const idIndex = headers?.indexOf('id') ?? -1; const metadataIndex = headers?.indexOf('metadata') ?? -1;
    documents = rows.map((row, index) => {
      let metadata: Record<string, unknown> = {};
      if (metadataIndex >= 0 && row[metadataIndex]) {
        try { metadata = JSON.parse(row[metadataIndex]) as Record<string, unknown>; }
        catch { metadata = { imported_metadata: row[metadataIndex] }; }
      }
      return { id: idIndex >= 0 && row[idIndex] ? row[idIndex] : `doc-${index + 1}`,
        text: row[textIndex] || '', metadata };
    });
  } else {
    documents = content.split(/\r?\n/).filter(Boolean).map((text, index) =>
      ({ id: `doc-${index + 1}`, text, metadata: {} }));
  }
  documents = documents.filter((document) => document.text.length > 0);
  if (!documents.length) throw new Error('No documents found');
  if (documents.length > 100) throw new Error('A request may contain at most 100 documents');
  return documents;
}

function Playground({ token }: { token: string }) {
  const [initialState] = useState(loadPlaygroundState);
  const [query, setQuery] = useState(initialState.query);
  const [documents, setDocuments] = useState<PlaygroundDocument[]>(initialState.documents);
  const [topN, setTopN] = useState(initialState.top_n); const [importError, setImportError] = useState('');
  const [dragged, setDragged] = useState<number | null>(null);
  const mutation = useMutation({ mutationFn: () => api<Record<string, unknown>>('admin/rerank', token, {
    method: 'POST', body: JSON.stringify({ query, documents, top_n: topN,
      return_documents: true, truncate: true }),
  }) });
  useEffect(() => {
    const timeout = window.setTimeout(() => {
      savePlaygroundState({ query, documents, top_n: topN });
    }, PLAYGROUND_SAVE_DELAY_MS);
    return () => window.clearTimeout(timeout);
  }, [query, documents, topN]);
  const move = (from: number, to: number) => {
    if (to < 0 || to >= documents.length || from === to) return;
    const reordered = [...documents]; const [document] = reordered.splice(from, 1);
    reordered.splice(to, 0, document); setDocuments(reordered);
  };
  return <section><header><div><h2>Rerank Playground</h2><p>{documents.length}/100 documents</p></div>
    <label className="file-button">Import JSON, CSV, or text<input aria-label="Import documents" type="file"
      accept=".json,.csv,.txt,text/plain,application/json,text/csv" onChange={async (event) => {
        const file = event.target.files?.[0]; if (!file) return;
        try { setDocuments(parseDocuments(file.name, await file.text())); setImportError(''); }
        catch (error) { setImportError(error instanceof Error ? error.message : 'Import failed'); }
      }} /></label></header>
    <label>Query<textarea maxLength={8000} value={query}
      onChange={(event) => setQuery(event.target.value)} /></label>
    {importError && <p className="error" role="alert">{importError}</p>}
    <div className="document-list">{documents.map((document, index) => <article key={`${document.id}-${index}`}
      draggable onDragStart={() => setDragged(index)} onDragOver={(event) => event.preventDefault()}
      onDrop={() => { if (dragged !== null) move(dragged, index); setDragged(null); }}>
      <span className="drag" title="Drag to reorder">⋮⋮</span>
      <label>ID<input aria-label={`Document ID ${index + 1}`} value={document.id}
        onChange={(event) => setDocuments(documents.map((item, itemIndex) =>
          itemIndex === index ? { ...item, id: event.target.value } : item))} /></label>
      <label>Text<textarea aria-label={`Document text ${index + 1}`} maxLength={20000} value={document.text}
        onChange={(event) => setDocuments(documents.map((item, itemIndex) =>
          itemIndex === index ? { ...item, text: event.target.value } : item))} /></label>
      <div className="document-actions"><button className="secondary" disabled={index === 0}
        onClick={() => move(index, index - 1)}>Move up</button><button className="secondary"
        disabled={index === documents.length - 1} onClick={() => move(index, index + 1)}>Move down</button>
        <button className="danger" disabled={documents.length === 1}
          onClick={() => setDocuments(documents.filter((_, itemIndex) => itemIndex !== index))}>Remove</button></div>
    </article>)}</div><div className="actions"><button onClick={() => setDocuments([...documents,
      { id: `doc-${documents.length + 1}`, text: '', metadata: {} }])}>Add document</button>
      <label>Top N<input aria-label="Top N" type="number" min="1" max="100" value={topN}
        onChange={(event) => setTopN(Number(event.target.value))} /></label>
      <button onClick={() => mutation.mutate()} disabled={mutation.isPending}>Run rerank</button>
      <button className="danger" onClick={() => {
        mutation.reset(); setQuery(''); setDocuments([]); setTopN(10); setImportError(''); setDragged(null);
        try { localStorage.removeItem(PLAYGROUND_STORAGE_KEY); } catch { /* Keep the UI clear. */ }
      }}>Clear Playground</button></div>
    {mutation.error && <p className="error" role="alert">{mutation.error.message}</p>}
    {mutation.data !== undefined && <pre>{JSON.stringify(mutation.data, null, 2)}</pre>}
  </section>;
}

type BatchItem = { key: number; query: string; documents: string; topN: number };

function BatchPlayground({ token }: { token: string }) {
  const [nextKey, setNextKey] = useState(2);
  const [items, setItems] = useState<BatchItem[]>([{
    key: 1, query: 'Kubernetes experience',
    documents: 'Production Kubernetes operations\nMicrosoft SQL Server administration', topN: 2,
  }]);
  const mutation = useMutation({ mutationFn: () => api<Record<string, unknown>>(
    'admin/rerank/batch', token, { method: 'POST', body: JSON.stringify({ requests: items.map((item) => ({
      query: item.query,
      documents: item.documents.split('\n').filter(Boolean).slice(0, 100).map((text, index) => ({
        id: `request-${item.key}-doc-${index + 1}`, text,
      })),
      top_n: item.topN, return_documents: true, truncate: true,
    })) }) },
  ) });
  const update = (key: number, patch: Partial<BatchItem>) => setItems(
    items.map((item) => item.key === key ? { ...item, ...patch } : item),
  );
  const totalPairs = items.reduce((total, item) =>
    total + item.documents.split('\n').filter(Boolean).length, 0);
  const exportResult = () => {
    if (!mutation.data) return;
    const blob = new Blob([JSON.stringify(mutation.data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url; anchor.download = 'reranker-batch-result.json'; anchor.click();
    URL.revokeObjectURL(url);
  };
  return <section><header><div><h2>Batch Playground</h2>
    <p>{items.length}/32 requests · {totalPairs} total pairs</p></div>
    <button disabled={items.length >= 32} onClick={() => {
      setItems([...items, { key: nextKey, query: '', documents: '', topN: 10 }]);
      setNextKey(nextKey + 1);
    }}>Add request</button></header><div className="run-list">{items.map((item, index) =>
      <article key={item.key}><header><strong>Request {index + 1}</strong>
        <button className="danger" disabled={items.length === 1}
          onClick={() => setItems(items.filter((candidate) => candidate.key !== item.key))}>Remove</button></header>
        <label>Query<input aria-label={`Batch query ${index + 1}`} maxLength={8000} value={item.query}
          onChange={(event) => update(item.key, { query: event.target.value })} /></label>
        <label>Documents - one per line<textarea aria-label={`Batch documents ${index + 1}`} rows={5}
          value={item.documents} onChange={(event) => update(item.key, { documents: event.target.value })} /></label>
        <label>Top N<input aria-label={`Batch top N ${index + 1}`} type="number" min="1" max="100"
          value={item.topN} onChange={(event) => update(item.key, { topN: Number(event.target.value) })} /></label>
      </article>)}</div><div className="actions"><button onClick={() => mutation.mutate()}
        disabled={mutation.isPending || totalPairs === 0}>Run batch</button>
      <button className="secondary" disabled={!mutation.data} onClick={exportResult}>Export JSON</button></div>
    {mutation.error && <p className="error" role="alert">{mutation.error.message}</p>}
    {mutation.data !== undefined && <pre>{JSON.stringify(mutation.data, null, 2)}</pre>}
  </section>;
}

type RuntimeState = {
  batch_size: number; max_concurrency: number; max_length: number;
  dynamic_batching: boolean; batch_window_ms: number; max_batch_pairs: number;
  request_timeout_seconds: number; default_top_n: number; queue_depth: number;
};

function RuntimePanel({ token }: { token: string }) {
  const client = useQueryClient();
  const runtime = useQuery({ queryKey: ['runtime'], queryFn: () => api<RuntimeState>('admin/runtime', token) });
  const [form, setForm] = useState<RuntimeState | null>(null);
  useEffect(() => { if (runtime.data && !form) setForm(runtime.data); }, [runtime.data, form]);
  const validate = useMutation({ mutationFn: (body: Partial<RuntimeState>) => api<{
    valid: boolean; memory_warning: boolean; restart_required: boolean;
  }>('admin/runtime/validate', token, { method: 'POST', body: JSON.stringify(body) }) });
  const save = useMutation({
    mutationFn: (body: Partial<RuntimeState>) => api('admin/runtime', token, {
      method: 'PATCH', body: JSON.stringify(body),
    }),
    onSuccess: () => client.invalidateQueries({ queryKey: ['runtime'] }),
  });
  if (!form) return <div className="skeleton">Loading…</div>;
  const numeric = (key: keyof RuntimeState, value: string) => setForm({ ...form, [key]: Number(value) });
  const payload = {
    batch_size: form.batch_size, max_concurrency: form.max_concurrency,
    max_length: form.max_length, dynamic_batching: form.dynamic_batching,
    batch_window_ms: form.batch_window_ms, max_batch_pairs: form.max_batch_pairs,
    request_timeout_seconds: form.request_timeout_seconds, default_top_n: form.default_top_n,
  };
  return <section><header><div><h2>Runtime</h2><p>Queue depth: {form.queue_depth}</p></div></header>
    <div className="form-grid">
      {(['batch_size', 'max_concurrency', 'max_length', 'batch_window_ms', 'max_batch_pairs',
        'request_timeout_seconds', 'default_top_n'] as const).map((key) =>
        <label key={key}>{key.replaceAll('_', ' ')}<input type="number" value={form[key]}
          onChange={(event) => numeric(key, event.target.value)} /></label>)}
      <label className="toggle">Dynamic batching<input type="checkbox" checked={form.dynamic_batching}
        onChange={(event) => setForm({ ...form, dynamic_batching: event.target.checked })} /></label>
    </div>
    <div className="actions"><button className="secondary" onClick={() => validate.mutate(payload)}>
      Validate configuration</button><button onClick={() => save.mutate(payload)}>Apply settings</button></div>
    {validate.data && <p className={validate.data.memory_warning ? 'warning' : 'success'}>
      {validate.data.memory_warning ? 'Valid, but memory risk is elevated.' : 'Configuration is valid.'}</p>}
    {(validate.error || save.error) && <p className="error" role="alert">
      {(validate.error || save.error)?.message}</p>}
  </section>;
}

type CacheState = { enabled: boolean; ttl_seconds: number; redis_available: boolean };

function CachePanel({ token }: { token: string }) {
  const client = useQueryClient();
  const cache = useQuery({ queryKey: ['cache'], queryFn: () => api<CacheState>('admin/cache', token) });
  const update = useMutation({
    mutationFn: (body: Partial<CacheState>) => api<CacheState>('admin/cache', token, {
      method: 'PATCH', body: JSON.stringify(body),
    }),
    onSuccess: () => client.invalidateQueries({ queryKey: ['cache'] }),
  });
  const clear = useMutation({
    mutationFn: () => api<{ deleted: number }>('admin/cache/clear', token, {
      method: 'POST', body: JSON.stringify({ confirm: 'CLEAR' }),
    }),
  });
  if (!cache.data) return <div className="skeleton">Loading…</div>;
  const data = cache.data;
  return <section><h2>Cache</h2><div className="cards">
    <article><span>Redis</span><strong>{data.redis_available ? 'Available' : 'Degraded'}</strong></article>
    <article><span>Cache</span><strong>{data.enabled ? 'Enabled' : 'Disabled'}</strong></article>
    <article><span>TTL</span><strong>{data.ttl_seconds}s</strong></article>
  </div><div className="form-grid"><label className="toggle">Cache enabled
    <input aria-label="Cache enabled" type="checkbox" checked={data.enabled}
      onChange={(event) => update.mutate({ enabled: event.target.checked })} /></label>
    <label>TTL seconds<input aria-label="TTL seconds" type="number" min="1" max="2592000"
      defaultValue={data.ttl_seconds} onBlur={(event) => update.mutate({ ttl_seconds: Number(event.target.value) })} /></label>
  </div><button className="danger" onClick={() => {
    if (window.confirm('Clear the entire reranker cache?')) clear.mutate();
  }}>Clear cache</button>
  {clear.data && <p className="success">Deleted {clear.data.deleted} keys.</p>}
  {(update.error || clear.error) && <p className="error" role="alert">{(update.error || clear.error)?.message}</p>}
  </section>;
}

type BenchmarkRun = {
  id: string; status: string; baseline: boolean; created_at: number;
  parameters: Record<string, unknown>; results?: Record<string, number> | null;
};

function BenchmarksPanel({ token }: { token: string }) {
  const client = useQueryClient();
  const [repetitions, setRepetitions] = useState(5);
  const runs = useQuery({
    queryKey: ['benchmarks'], queryFn: () => api<{ items: BenchmarkRun[] }>('admin/benchmarks', token),
    refetchInterval: 1000,
  });
  const start = useMutation({
    mutationFn: () => api<BenchmarkRun>('admin/benchmarks', token, {
      method: 'POST', body: JSON.stringify({ mode: 'low_priority', repetitions,
        warmup_count: 1, document_count: 2, multilingual: true }),
    }),
    onSuccess: () => client.invalidateQueries({ queryKey: ['benchmarks'] }),
  });
  const baseline = useMutation({
    mutationFn: (id: string) => api(`admin/benchmarks/${id}/baseline`, token, { method: 'POST' }),
    onSuccess: () => client.invalidateQueries({ queryKey: ['benchmarks'] }),
  });
  const remove = useMutation({
    mutationFn: (id: string) => api(`admin/benchmarks/${id}`, token, { method: 'DELETE' }),
    onSuccess: () => client.invalidateQueries({ queryKey: ['benchmarks'] }),
  });
  return <section><header><div><h2>Benchmarks</h2><p>Runs use the built-in multilingual dataset.</p></div>
    <div className="actions"><label>Repetitions<input aria-label="Benchmark repetitions" type="number"
      min="1" max="100" value={repetitions}
      onChange={(event) => setRepetitions(Number(event.target.value))} /></label>
      <button onClick={() => start.mutate()} disabled={start.isPending}>Run benchmark</button></div></header>
    {start.error && <p className="error" role="alert">{start.error.message}</p>}
    <div className="run-list">{runs.data?.items.map((run) => <article key={run.id}>
      <header><strong>{run.id.slice(0, 8)}</strong><span>{run.status}{run.baseline ? ' · baseline' : ''}</span></header>
      {run.results && <div className="cards">
        {['p50_ms', 'p95_ms', 'p99_ms', 'pairs_per_second'].map((metric) => <div key={metric}>
          <span>{metric}</span><strong>{run.results?.[metric]?.toFixed(2)}</strong></div>)}
      </div>}
      <div className="actions"><button className="secondary" disabled={run.status !== 'completed'}
        onClick={() => baseline.mutate(run.id)}>Set baseline</button>
        <button className="danger" onClick={() => {
          if (window.confirm('Delete this benchmark run?')) remove.mutate(run.id);
        }}>Delete</button></div>
    </article>)}</div>
  </section>;
}

type DashboardState = {
  model: { name: string; revision: string; ready: boolean; device: string;
    backend: { backend: string; active_provider?: string; fallback_reason?: string | null;
      gpu_name?: string | null; available_providers?: string[] } };
  redis: { available: boolean };
  resources: { cpu_percent: number; ram_percent: number; uptime_seconds: number };
};

function DashboardPanel({ token }: { token: string }) {
  const dashboard = useQuery({ queryKey: ['dashboard'],
    queryFn: () => api<DashboardState>('admin/dashboard', token), refetchInterval: 10000 });
  const metrics = useQuery({ queryKey: ['timeseries'],
    queryFn: () => api<{ points: Array<Record<string, number>> }>(
      'admin/metrics/timeseries?period_seconds=3600&bucket_seconds=60', token),
    refetchInterval: 10000 });
  if (!dashboard.data) return <div className="skeleton">Loading…</div>;
  const data = dashboard.data;
  return <section><h2>Dashboard</h2><div className="cards">
    <article><span>Model</span><strong>{data.model.ready ? 'Ready' : 'Not ready'}</strong></article>
    <article><span>Backend</span><strong>{data.model.backend.backend}</strong></article>
    <article><span>Provider</span><strong>{data.model.backend.active_provider || data.model.device}</strong>
      {data.model.backend.fallback_reason && <small>{data.model.backend.fallback_reason}</small>}</article>
    <article><span>GPU</span><strong>{data.model.backend.gpu_name || 'Not active'}</strong></article>
    <article><span>Redis</span><strong>{data.redis.available ? 'Available' : 'Degraded'}</strong></article>
    <article><span>CPU</span><strong>{data.resources.cpu_percent}%</strong></article>
    <article><span>RAM</span><strong>{data.resources.ram_percent}%</strong></article>
  </div><h3>Last hour</h3><div className="metric-table">
    {metrics.data?.points.length ? metrics.data.points.map((point) => <div key={point.timestamp}>
      <time>{new Date(point.timestamp * 1000).toLocaleTimeString()}</time>
      <span>{point.requests} requests</span><span>{point.latency_p95_ms.toFixed(1)}ms p95</span>
      <span>{point.cache_hits} cache hits</span></div>) : <p>No requests in this period.</p>}
  </div></section>;
}

type ModelInfo = {
  name: string; revision: string | null; status: string; loaded: boolean;
  device_support: string[]; max_length: number; estimated_memory_bytes: number;
  average_latency_ms: number | null;
};

type ModelsResponse = {
  active_model: string;
  candidate: Record<string, unknown> | null;
  rollback_available: boolean;
  models: ModelInfo[];
};

type ModelCandidateResult = {
  name?: string;
  revision?: string | null;
  requested_revision?: string;
  valid?: boolean;
  controlled_restart_required?: boolean;
  status?: string;
  error?: string | null;
};

function ModelsPanel({ token }: { token: string }) {
  const queryClient = useQueryClient();
  const [name, setName] = useState('');
  const [revision, setRevision] = useState('');

  const query = useQuery({
    queryKey: ['models'],
    queryFn: () => api<ModelsResponse>('admin/models', token),
    retry: false,
  });

  // Do not return invalidateQueries() from mutation callbacks. React Query waits
  // for a returned Promise, which can leave the mutation visually pending while
  // a refetch is slow or unavailable.
  const refresh = () => {
    void queryClient.invalidateQueries({ queryKey: ['models'] });
    void queryClient.invalidateQueries({ queryKey: ['runtime'] });
    void queryClient.invalidateQueries({ queryKey: ['system-health'] });
    void queryClient.invalidateQueries({ queryKey: ['dashboard'] });
  };

  const candidateName = (name.trim() || query.data?.active_model || '').trim();

  const candidateBody = (): { name: string; revision?: string } => {
    const body: { name: string; revision?: string } = { name: candidateName };
    const pinnedRevision = revision.trim();
    // Empty revision intentionally means "resolve current Hugging Face main".
    if (pinnedRevision) body.revision = pinnedRevision;
    return body;
  };

  const check = useMutation({
    mutationFn: () => api<ModelCandidateResult>('admin/models/check', token, {
      method: 'POST',
      body: JSON.stringify(candidateBody()),
    }),
    onSuccess: (result) => {
      // Backend resolves main/tag/short SHA to an immutable full SHA.
      if (typeof result.revision === 'string' && result.revision) {
        setRevision(result.revision);
      }
    },
  });

  const load = useMutation({
    mutationFn: () => api<ModelCandidateResult>('admin/models/load', token, {
      method: 'POST',
      body: JSON.stringify(candidateBody()),
    }),
    onSuccess: (result) => {
      if (typeof result.revision === 'string' && result.revision) {
        setRevision(result.revision);
      }
      refresh();
    },
    onError: () => {
      // A failed load may still have updated candidate/error state in runtime.
      refresh();
    },
  });

  const activate = useMutation({
    mutationFn: () => api<ModelCandidateResult>(
      'admin/models/activate',
      token,
      { method: 'POST', body: JSON.stringify({ confirm: 'ACTIVATE' }) },
    ),
    onSuccess: (result) => {
      if (typeof result.revision === 'string' && result.revision) {
        setRevision(result.revision);
      }
      refresh();
    },
    onError: refresh,
  });

  const rollback = useMutation({
    mutationFn: () => api<ModelCandidateResult>(
      'admin/runtime/rollback',
      token,
      { method: 'POST' },
    ),
    onSuccess: refresh,
    onError: refresh,
  });

  const busy = check.isPending || load.isPending || activate.isPending || rollback.isPending;
  const candidateStatus = typeof query.data?.candidate?.status === 'string'
    ? query.data.candidate.status
    : null;
  const candidateReady = Boolean(query.data?.candidate)
    && (candidateStatus === null || candidateStatus === 'ready');

  const operationError = check.error || load.error || activate.error || rollback.error;

  if (query.isLoading) return <div className="skeleton">Loading…</div>;

  // Previously `!query.data` always rendered "Loading…" even after a failed
  // request, producing an endless loading indicator.
  if (query.error) {
    return <section>
      <h2>Models</h2>
      <p className="error" role="alert">{query.error.message}</p>
      <button className="secondary" onClick={() => { void query.refetch(); }}>Retry</button>
    </section>;
  }

  if (!query.data) {
    return <section>
      <h2>Models</h2>
      <p className="error" role="alert">Model state is unavailable.</p>
      <button className="secondary" onClick={() => { void query.refetch(); }}>Retry</button>
    </section>;
  }

  return <section>
    <h2>Models</h2>

    <div className="form-grid">
      <label>
        Candidate model
        <input
          aria-label="Candidate model"
          value={name || query.data.active_model}
          onChange={(event) => {
            setName(event.target.value);
            setRevision('');
            check.reset();
            load.reset();
          }}
        />
      </label>

      <label>
        Immutable revision
        <input
          aria-label="Immutable revision"
          value={revision}
          onChange={(event) => {
            setRevision(event.target.value);
            check.reset();
            load.reset();
          }}
          placeholder="Auto: current Hugging Face main"
        />
      </label>
    </div>

    <p>
      Leave revision empty to resolve the current Hugging Face <code>main</code>.
      Check pins it to an immutable commit SHA before loading.
    </p>

    <div className="actions">
      <button
        className="secondary"
        disabled={!candidateName || busy}
        onClick={() => check.mutate()}
      >
        {check.isPending ? 'Checking…' : 'Check'}
      </button>

      <button
        className="secondary"
        disabled={!candidateName || busy}
        onClick={() => load.mutate()}
      >
        {load.isPending ? 'Loading candidate…' : 'Load candidate'}
      </button>

      <button
        disabled={!candidateReady || busy}
        onClick={() => {
          if (window.confirm('Activate the warmed candidate model?')) activate.mutate();
        }}
      >
        {activate.isPending ? 'Activating…' : 'Activate candidate'}
      </button>

      <button
        className="danger"
        disabled={!query.data.rollback_available || busy}
        onClick={() => rollback.mutate()}
      >
        {rollback.isPending ? 'Rolling back…' : 'Rollback model'}
      </button>
    </div>

    {check.data && <p className={check.data.valid ? 'success' : 'error'}>
      {check.data.valid
        ? <><span>Candidate is valid.</span>{' '}
          Pinned revision: {check.data.revision || 'unavailable'}</>
        : `Candidate cannot be loaded${check.data.error ? `: ${check.data.error}` : '.'}`}
    </p>}

    {load.data?.controlled_restart_required === true &&
      <p>Insufficient memory: controlled restart required.</p>}

    {candidateStatus && <p>Candidate status: <strong>{candidateStatus}</strong></p>}

    {operationError &&
      <p className="error" role="alert">
        {operationError instanceof Error ? operationError.message : String(operationError)}
      </p>}

    <div className="run-list">
      {query.data.models.map((model) =>
        <article key={model.name}>
          <header>
            <strong>{model.name}</strong>
            <span>{model.loaded ? 'Active · ready' : model.status}</span>
          </header>
          <dl className="details">
            <div><dt>Revision</dt><dd>{model.revision || 'Not loaded'}</dd></div>
            <div><dt>Devices</dt><dd>{model.device_support.join(', ')}</dd></div>
            <div><dt>Max length</dt><dd>{model.max_length}</dd></div>
            <div><dt>Estimated memory</dt><dd>{(model.estimated_memory_bytes / 1e9).toFixed(1)} GB</dd></div>
            <div><dt>Average latency</dt><dd>{model.average_latency_ms?.toFixed(1) || 'No data'} ms</dd></div>
          </dl>
        </article>)}
    </div>
  </section>;
}

type RequestRecord = {
  request_id: string; correlation_id: string; timestamp: number; documents_count: number;
  model: string; device: string; latency_ms: number; cache_hits: number; status: string;
  truncation_count: number;
};

function RequestsPanel({ token }: { token: string }) {
  const [page, setPage] = useState(1);
  const query = useQuery({ queryKey: ['requests', page],
    queryFn: () => api<{ items: RequestRecord[]; total: number; size: number }>(
      `admin/requests?page=${page}&size=20`, token) });
  if (!query.data) return <div className="skeleton">Loading…</div>;
  return <section><header><h2>Requests</h2><span>{query.data.total} retained technical records</span></header>
    <div className="table-wrap"><table><thead><tr><th>Request</th><th>Correlation</th><th>Documents</th>
      <th>Model / device</th><th>Latency</th><th>Cache</th><th>Status</th></tr></thead><tbody>
      {query.data.items.map((item) => <tr key={`${item.request_id}-${item.timestamp}`}>
        <td title={item.request_id}>{item.request_id.slice(0, 8)}</td>
        <td title={item.correlation_id}>{item.correlation_id.slice(0, 12)}</td>
        <td>{item.documents_count}</td><td>{item.model}<br /><small>{item.device}</small></td>
        <td>{item.latency_ms}ms</td><td>{item.cache_hits}</td><td>{item.status}</td></tr>)}
    </tbody></table></div><div className="actions"><button className="secondary" disabled={page === 1}
      onClick={() => setPage(page - 1)}>Previous</button><span>Page {page}</span>
      <button className="secondary" disabled={page * query.data.size >= query.data.total}
        onClick={() => setPage(page + 1)}>Next</button></div></section>;
}

function SystemHealthPanel({ token }: { token: string }) {
  const health = useQuery({ queryKey: ['system-health'],
    queryFn: () => api<Record<string, string | boolean>>('admin/system/health', token),
    refetchInterval: 5000 });
  const resources = useQuery({ queryKey: ['system-resources'],
    queryFn: () => api<Record<string, number | boolean | null>>('admin/system/resources', token),
    refetchInterval: 5000 });
  if (!health.data || !resources.data) return <div className="skeleton">Loading…</div>;
  return <section><header><h2>System Health</h2><button className="secondary" onClick={() => {
    health.refetch(); resources.refetch();
  }}>Refresh</button></header><div className="cards">
    {Object.entries(health.data).map(([name, value]) => <article key={name}><span>{name}</span>
      <strong>{String(value)}</strong></article>)}
  </div><h3>Resources</h3><dl className="details">{Object.entries(resources.data).map(([name, value]) =>
    <div key={name}><dt>{name.replaceAll('_', ' ')}</dt><dd>{String(value ?? 'Unavailable')}</dd></div>)}</dl>
  </section>;
}

function SettingsPanel({ token }: { token: string }) {
  const runtime = useQuery({ queryKey: ['settings-runtime'],
    queryFn: () => api<RuntimeState>('admin/runtime', token) });
  const cache = useQuery({ queryKey: ['settings-cache'],
    queryFn: () => api<CacheState>('admin/cache', token) });
  const validate = useMutation({ mutationFn: () => {
    if (!runtime.data) throw new Error('Runtime settings are not loaded');
    const payload = {
      batch_size: runtime.data.batch_size, max_concurrency: runtime.data.max_concurrency,
      max_length: runtime.data.max_length, dynamic_batching: runtime.data.dynamic_batching,
      batch_window_ms: runtime.data.batch_window_ms, max_batch_pairs: runtime.data.max_batch_pairs,
      request_timeout_seconds: runtime.data.request_timeout_seconds,
      default_top_n: runtime.data.default_top_n,
    };
    return api<{ valid: boolean; memory_warning: boolean }>('admin/runtime/validate', token, {
      method: 'POST', body: JSON.stringify(payload),
    });
  } });
  if (!runtime.data || !cache.data) return <div className="skeleton">Loading…</div>;
  return <section><h2>Settings</h2><p>Secrets are managed externally and are never exposed here.</p>
    <div className="cards"><article><span>Batch size</span><strong>{runtime.data.batch_size}</strong></article>
      <article><span>Concurrency</span><strong>{runtime.data.max_concurrency}</strong></article>
      <article><span>Max length</span><strong>{runtime.data.max_length}</strong></article>
      <article><span>Cache TTL</span><strong>{cache.data.ttl_seconds}s</strong></article></div>
    <button onClick={() => validate.mutate()}>Validate configuration</button>
    {validate.data && <p className={validate.data.memory_warning ? 'warning' : 'success'}>
      {validate.data.memory_warning ? 'Valid with elevated memory risk.' : 'Configuration is valid.'}</p>}
    {validate.error && <p className="error" role="alert">{validate.error.message}</p>}
  </section>;
}

type AuditRecord = { id: string; timestamp: number; action: string; details: Record<string, unknown> };

function AuditLogPanel({ token }: { token: string }) {
  const [page, setPage] = useState(1);
  const query = useQuery({ queryKey: ['audit-log', page], queryFn: () =>
    api<{ items: AuditRecord[]; total: number }>(`admin/audit-log?page=${page}&size=25`, token) });
  if (!query.data) return <div className="skeleton">Loading…</div>;
  return <section><header><h2>Audit Log</h2><span>{query.data.total} events</span></header>
    <div className="table-wrap"><table><thead><tr><th>Time</th><th>Action</th><th>Details</th></tr></thead>
      <tbody>{query.data.items.map((record) => <tr key={record.id}>
        <td>{new Date(record.timestamp * 1000).toLocaleString()}</td><td>{record.action}</td>
        <td><code>{JSON.stringify(record.details)}</code></td></tr>)}</tbody></table></div>
    <div className="actions"><button className="secondary" disabled={page === 1}
      onClick={() => setPage(page - 1)}>Previous</button><span>Page {page}</span>
      <button className="secondary" disabled={page * 25 >= query.data.total}
        onClick={() => setPage(page + 1)}>Next</button></div></section>;
}

function DataView({ section, token }: { section: Section; token: string }) {
  const endpoint = `admin/${paths[section]}`;
  const query = useQuery({ queryKey: [endpoint], queryFn: () => api(endpoint, token),
    refetchInterval: section === 'Dashboard' ? 10000 : false });
  if (query.isLoading) return <div className="skeleton">Loading…</div>;
  if (query.error) return <div className="error" role="alert">{query.error.message}</div>;
  return <section><header><h2>{section}</h2><button className="secondary"
    onClick={() => query.refetch()}>Refresh</button></header><pre>{JSON.stringify(query.data, null, 2)}</pre></section>;
}

export function App() {
  const [token, setToken] = useState(() => sessionStorage.getItem('adminToken') || '');
  const [section, setSection] = useState<Section>(() => {
    const hash = decodeURIComponent(location.hash.slice(1));
    return sections.includes(hash as Section) ? hash as Section : 'Dashboard';
  });
  useEffect(() => { location.hash = section; }, [section]);
  if (!token) return <Login onLogin={(value) => {
    sessionStorage.setItem('adminToken', value);
    setToken(value);
  }} />;
  let content: React.ReactNode;
  if (section === 'Dashboard') content = <DashboardPanel token={token} />;
  else if (section === 'Rerank Playground') content = <Playground token={token} />;
  else if (section === 'Batch Playground') content = <BatchPlayground token={token} />;
  else if (section === 'Runtime') content = <RuntimePanel token={token} />;
  else if (section === 'Cache') content = <CachePanel token={token} />;
  else if (section === 'Benchmarks') content = <BenchmarksPanel token={token} />;
  else if (section === 'Models') content = <ModelsPanel token={token} />;
  else if (section === 'Requests') content = <RequestsPanel token={token} />;
  else if (section === 'System Health') content = <SystemHealthPanel token={token} />;
  else if (section === 'Settings') content = <SettingsPanel token={token} />;
  else if (section === 'Audit Log') content = <AuditLogPanel token={token} />;
  else content = <DataView section={section} token={token} />;
  return <div className="shell"><aside><div className="brand">RR <span>Console</span></div>
    {sections.map((item) => <button className={item === section ? 'active' : ''}
      onClick={() => setSection(item)} key={item}>{item}</button>)}
    <button onClick={() => { sessionStorage.clear(); setToken(''); }}>Log out</button>
  </aside><main>{content}</main></div>;
}

const root = document.getElementById('root');
if (root) createRoot(root).render(<React.StrictMode><QueryClientProvider client={new QueryClient()}>
  <App /></QueryClientProvider></React.StrictMode>);
