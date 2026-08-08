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
  if (section === 'Rerank Playground') content = <Playground token={token} />;
  else if (section === 'Batch Playground') content = <Playground token={token} batch />;
  else if (section === 'Runtime') content = <RuntimePanel token={token} />;
  else if (section === 'Cache') content = <CachePanel token={token} />;
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
