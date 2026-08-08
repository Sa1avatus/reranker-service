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
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error?.message || response.statusText);
  return payload as T;
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

function Playground({ token, batch = false }: { token: string; batch?: boolean }) {
  const [query, setQuery] = useState("Describe the candidate's Kubernetes experience");
  const [documents, setDocuments] = useState(
    'Basic conceptual knowledge of Kubernetes. No hands-on production experience.\n' +
    'Production experience with Docker and Docker Compose.',
  );
  const mutation = useMutation({ mutationFn: async () => {
    const request = {
      query,
      documents: documents.split('\n').filter(Boolean).map((text, index) => (
        { id: `doc-${index + 1}`, text }
      )),
      top_n: 10, return_documents: true, truncate: true,
    };
    return api(batch ? 'admin/rerank/batch' : 'admin/rerank', token, {
      method: 'POST', body: JSON.stringify(batch ? { requests: [request] } : request),
    });
  }});
  return <section><h2>{batch ? 'Batch' : 'Rerank'} Playground</h2>
    <label>Query<textarea value={query} onChange={(event) => setQuery(event.target.value)} /></label>
    <label>Documents - one per line<textarea rows={8} value={documents}
      onChange={(event) => setDocuments(event.target.value)} /></label>
    <button onClick={() => mutation.mutate()} disabled={mutation.isPending}>Run rerank</button>
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
  model: { name: string; revision: string; ready: boolean; device: string };
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

function ModelsPanel({ token }: { token: string }) {
  const query = useQuery({ queryKey: ['models'],
    queryFn: () => api<{ active_model: string; models: ModelInfo[] }>('admin/models', token) });
  if (!query.data) return <div className="skeleton">Loading…</div>;
  return <section><h2>Models</h2><div className="run-list">{query.data.models.map((model) =>
    <article key={model.name}><header><strong>{model.name}</strong>
      <span>{model.loaded ? 'Active · ready' : model.status}</span></header>
      <dl className="details"><div><dt>Revision</dt><dd>{model.revision || 'Not loaded'}</dd></div>
        <div><dt>Devices</dt><dd>{model.device_support.join(', ')}</dd></div>
        <div><dt>Max length</dt><dd>{model.max_length}</dd></div>
        <div><dt>Estimated memory</dt><dd>{(model.estimated_memory_bytes / 1e9).toFixed(1)} GB</dd></div>
        <div><dt>Average latency</dt><dd>{model.average_latency_ms?.toFixed(1) || 'No data'} ms</dd></div>
      </dl></article>)}</div></section>;
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
