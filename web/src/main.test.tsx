import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import {
  App,
  loadPlaygroundState,
  Login,
  parseDocuments,
  PLAYGROUND_STORAGE_KEY,
  PLAYGROUND_REMEMBER_KEY,
  resolveBackendSelection,
  savePlaygroundState,
} from './main';

describe('backend selection', () => {
  const registry = {
    default_backend: 'legacy',
    backends: [
      { id: 'jina', name: 'Jina', backend: 'jina_listwise', available: true },
      { id: 'legacy', name: 'Legacy', backend: 'legacy_cross_encoder', available: true },
      { id: 'down', name: 'Down', backend: 'onnx_pairwise', available: false },
    ],
    model_map: {},
  };

  it('uses the server default when there is no valid saved selection', () => {
    expect(resolveBackendSelection('', registry)).toBe('legacy');
    expect(resolveBackendSelection('unknown', registry)).toBe('legacy');
    expect(resolveBackendSelection('down', registry)).toBe('legacy');
  });

  it('preserves an explicitly saved available backend', () => {
    expect(resolveBackendSelection('jina', registry)).toBe('jina');
  });

  it('falls back to the first available backend or no header', () => {
    expect(resolveBackendSelection('', { ...registry, default_backend: 'missing' })).toBe('jina');
    expect(resolveBackendSelection('', {
      ...registry,
      default_backend: 'down',
      backends: registry.backends.map((backend) => ({ ...backend, available: false })),
    })).toBe('');
  });
});

describe('document imports', () => {
  it('preserves JSON document IDs and metadata', () => {
    const documents = parseDocuments('documents.json', JSON.stringify([
      { id: 'original-id', text: 'Kubernetes', metadata: { collection: 'profile' } },
    ]));
    expect(documents[0]).toEqual({
      id: 'original-id', text: 'Kubernetes', metadata: { collection: 'profile' },
    });
  });

  it('parses quoted CSV fields', () => {
    const documents = parseDocuments('documents.csv',
      'id,text,metadata\nrow-1,"Python, FastAPI","{""language"":""en""}"');
    expect(documents[0].id).toBe('row-1');
    expect(documents[0].text).toBe('Python, FastAPI');
    expect(documents[0].metadata).toEqual({ language: 'en' });
  });
});

describe('authentication', () => {
  it('submits the admin token without rendering it as plain text', () => {
    const onLogin = vi.fn();
    render(<Login onLogin={onLogin} />);
    const input = screen.getByLabelText('Admin token') as HTMLInputElement;
    expect(input.type).toBe('password');
    fireEvent.change(input, { target: { value: 'admin-secret' } });
    fireEvent.click(screen.getByRole('button', { name: 'Sign in' }));
    expect(onLogin).toHaveBeenCalledWith('admin-secret');
  });

  it('supports logout and removes the session token', () => {
    sessionStorage.setItem('adminToken', 'admin-secret');
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => ({}) }));
    render(<QueryClientProvider client={new QueryClient()}><App /></QueryClientProvider>);
    fireEvent.click(screen.getByRole('button', { name: 'Log out' }));
    expect(sessionStorage.getItem('adminToken')).toBeNull();
    expect(screen.getByRole('button', { name: 'Sign in' })).toBeTruthy();
    vi.unstubAllGlobals();
  });
});

describe('Rerank Playground persistence', () => {
  beforeEach(() => localStorage.setItem(PLAYGROUND_REMEMBER_KEY, 'true'));

  const renderApp = () => render(
    <QueryClientProvider client={new QueryClient()}><App /></QueryClientProvider>,
  );

  const openPlayground = () => {
    sessionStorage.setItem('adminToken', 'admin-secret');
    location.hash = 'Rerank%20Playground';
  };

  it('autosaves and restores query, ordered documents, and top_n after a reload', async () => {
    openPlayground();
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => ({}) }));
    const firstRender = renderApp();

    fireEvent.change(screen.getByLabelText('Query'), { target: { value: 'Persistent query' } });
    fireEvent.change(screen.getByLabelText('Document ID 1'), { target: { value: 'first' } });
    fireEvent.change(screen.getByLabelText('Document text 1'), { target: { value: 'First text' } });
    fireEvent.change(screen.getByLabelText('Document ID 2'), { target: { value: 'second' } });
    fireEvent.change(screen.getByLabelText('Document text 2'), { target: { value: 'Second text' } });
    fireEvent.click(screen.getAllByRole('button', { name: 'Move down' })[0]);
    fireEvent.change(screen.getByLabelText('Top N'), { target: { value: '2' } });

    await waitFor(() => {
      const saved = JSON.parse(localStorage.getItem(PLAYGROUND_STORAGE_KEY) || '{}');
      expect(saved.query).toBe('Persistent query');
      expect(saved.documents.map((document: { id: string }) => document.id)).toEqual(['second', 'first']);
      expect(saved.top_n).toBe(2);
    });
    firstRender.unmount();

    renderApp();
    expect((screen.getByLabelText('Query') as HTMLTextAreaElement).value).toBe('Persistent query');
    expect((screen.getByLabelText('Document ID 1') as HTMLInputElement).value).toBe('second');
    expect((screen.getByLabelText('Document text 1') as HTMLTextAreaElement).value).toBe('Second text');
    expect((screen.getByLabelText('Document ID 2') as HTMLInputElement).value).toBe('first');
    expect((screen.getByLabelText('Document text 2') as HTMLTextAreaElement).value).toBe('First text');
    expect((screen.getByLabelText('Top N') as HTMLInputElement).value).toBe('2');
    vi.unstubAllGlobals();
  });

  it('keeps restored documents and metadata after rerank and active model changes', async () => {
    savePlaygroundState({
      query: 'Model-independent query',
      documents: [
        { id: 'doc-b', text: 'Second', metadata: { order: 2 } },
        { id: 'doc-a', text: 'First', metadata: { order: 1 } },
      ],
      top_n: 2,
    });
    openPlayground();
    const fetchMock = vi.fn().mockImplementation(async (url: string, options?: RequestInit) => {
      if (url.endsWith('/admin/models') && !options?.method) return { ok: true, json: async () => ({
        active_model: 'active-model', candidate: { name: 'candidate-model' }, rollback_available: false,
        models: [{ name: 'active-model', revision: 'abcdef1', status: 'ready', loaded: true,
          device_support: ['cpu'], max_length: 1024, estimated_memory_bytes: 1000,
          average_latency_ms: 1 }],
      }) };
      return { ok: true, json: async () => ({ results: [] }) };
    });
    vi.stubGlobal('fetch', fetchMock);
    const confirm = vi.spyOn(window, 'confirm').mockReturnValue(true);
    renderApp();

    fireEvent.click(screen.getByRole('button', { name: 'Run rerank' }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith('/v1/admin/rerank',
      expect.objectContaining({ method: 'POST' })));
    const rerankCall = fetchMock.mock.calls.find(([url]) => url === '/v1/admin/rerank');
    expect(JSON.parse(rerankCall?.[1]?.body as string).documents).toEqual([
      { id: 'doc-b', text: 'Second', metadata: { order: 2 } },
      { id: 'doc-a', text: 'First', metadata: { order: 1 } },
    ]);

    fireEvent.click(screen.getByRole('button', { name: 'Models' }));
    await screen.findByDisplayValue('active-model');
    fireEvent.click(screen.getByRole('button', { name: 'Activate candidate' }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith('/v1/admin/models/activate',
      expect.objectContaining({ method: 'POST' })));
    fireEvent.click(screen.getByRole('button', { name: 'Rerank Playground' }));
    expect((screen.getByLabelText('Query') as HTMLTextAreaElement).value).toBe('Model-independent query');
    expect((screen.getByLabelText('Document ID 1') as HTMLInputElement).value).toBe('doc-b');
    expect((screen.getByLabelText('Document ID 2') as HTMLInputElement).value).toBe('doc-a');
    expect(localStorage.getItem(PLAYGROUND_STORAGE_KEY)).not.toBeNull();
    confirm.mockRestore();
    vi.unstubAllGlobals();
  });

  it('clears the UI and leaves no persisted state after the debounce', async () => {
    savePlaygroundState({
      query: 'Clear me', documents: [{ id: 'doc-1', text: 'Text', metadata: {} }], top_n: 1,
    });
    openPlayground();
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => ({}) }));
    renderApp();

    fireEvent.click(screen.getByRole('button', { name: 'Clear Playground' }));
    expect((screen.getByLabelText('Query') as HTMLTextAreaElement).value).toBe('');
    expect(screen.queryByLabelText('Document ID 1')).toBeNull();
    expect((screen.getByLabelText('Top N') as HTMLInputElement).value).toBe('10');
    expect(localStorage.getItem(PLAYGROUND_STORAGE_KEY)).toBeNull();
    await new Promise((resolve) => window.setTimeout(resolve, 350));
    expect(localStorage.getItem(PLAYGROUND_STORAGE_KEY)).toBeNull();
    vi.unstubAllGlobals();
  });

  it('falls back safely for corrupt JSON and unknown schema versions', () => {
    localStorage.setItem(PLAYGROUND_STORAGE_KEY, '{broken');
    expect(() => loadPlaygroundState()).not.toThrow();
    expect(loadPlaygroundState().version).toBe(1);

    localStorage.setItem(PLAYGROUND_STORAGE_KEY, JSON.stringify({ version: 99, query: 'stale' }));
    openPlayground();
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => ({}) }));
    renderApp();
    expect(screen.getByRole('heading', { name: 'Rerank Playground' })).toBeTruthy();
    expect((screen.getByLabelText('Query') as HTMLTextAreaElement).value)
      .toBe("Describe the candidate's Kubernetes experience");
    vi.unstubAllGlobals();
  });

  it('restores inputs by default when remember key is absent', () => {
    localStorage.setItem(PLAYGROUND_STORAGE_KEY, JSON.stringify({
      version: 1, query: 'private', documents: [], top_n: 10,
    }));
    localStorage.removeItem(PLAYGROUND_REMEMBER_KEY);
    expect(loadPlaygroundState().query).toBe('private');
  });

  it('does not restore or retain inputs when explicitly opted out', () => {
    localStorage.setItem(PLAYGROUND_STORAGE_KEY, JSON.stringify({
      version: 1, query: 'private', documents: [], top_n: 10,
    }));
    localStorage.setItem(PLAYGROUND_REMEMBER_KEY, 'false');
    expect(loadPlaygroundState().query).not.toBe('private');
    expect(localStorage.getItem(PLAYGROUND_STORAGE_KEY)).toBeNull();
  });

  it('serializes only allowlisted Playground fields and never secrets', () => {
    const stateWithSecrets = {
      query: 'Safe query', documents: [{ id: 'doc-1', text: 'Safe text', metadata: {} }], top_n: 1,
      api_key: 'service-secret', admin_token: 'admin-secret', Authorization: 'Bearer secret',
      results: [{ score: 1 }],
    };
    savePlaygroundState(stateWithSecrets);
    const raw = localStorage.getItem(PLAYGROUND_STORAGE_KEY) || '';
    expect(Object.keys(JSON.parse(raw)).sort()).toEqual(['documents', 'query', 'top_n', 'version']);
    expect(raw).not.toContain('service-secret');
    expect(raw).not.toContain('admin-secret');
    expect(raw).not.toContain('Bearer secret');
    expect(raw).not.toContain('score');
  });
});

describe('administrative controls', () => {
  it('validates and applies runtime settings through the backend', async () => {
    sessionStorage.setItem('adminToken', 'admin-secret');
    const fetchMock = vi.fn().mockImplementation(async (url: string, options?: RequestInit) => {
      if (url.endsWith('/admin/runtime') && !options?.method) return { ok: true, json: async () => ({
        batch_size: 16, max_concurrency: 2, max_length: 1024, dynamic_batching: true,
        batch_window_ms: 10, max_batch_pairs: 128, request_timeout_seconds: 15,
        default_top_n: 10, queue_depth: 0,
      }) };
      if (url.endsWith('/validate')) return { ok: true, json: async () => ({ valid: true, memory_warning: false }) };
      return { ok: true, json: async () => ({ valid: true }) };
    });
    vi.stubGlobal('fetch', fetchMock);
    render(<QueryClientProvider client={new QueryClient()}><App /></QueryClientProvider>);
    fireEvent.click(screen.getByRole('button', { name: 'Runtime' }));
    await screen.findByDisplayValue('16');
    fireEvent.click(screen.getByRole('button', { name: 'Validate configuration' }));
    await screen.findByText('Configuration is valid.');
    fireEvent.click(screen.getByRole('button', { name: 'Apply settings' }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith('/v1/admin/runtime',
      expect.objectContaining({ method: 'PATCH' })));
    vi.unstubAllGlobals();
  });

  it('requires confirmation before clearing cache', async () => {
    sessionStorage.setItem('adminToken', 'admin-secret');
    const confirm = vi.spyOn(window, 'confirm').mockReturnValue(false);
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => ({
      enabled: true, ttl_seconds: 120, redis_available: true,
    }) });
    vi.stubGlobal('fetch', fetchMock);
    render(<QueryClientProvider client={new QueryClient()}><App /></QueryClientProvider>);
    fireEvent.click(screen.getByRole('button', { name: 'Cache' }));
    await screen.findByText('Available');
    fireEvent.click(screen.getByRole('button', { name: 'Clear cache' }));
    expect(confirm).toHaveBeenCalledOnce();
    expect(fetchMock).not.toHaveBeenCalledWith('/v1/admin/cache/clear', expect.anything());
    confirm.mockRestore();
    vi.unstubAllGlobals();
  });

  it('starts a low-priority multilingual benchmark', async () => {
    sessionStorage.setItem('adminToken', 'admin-secret');
    const fetchMock = vi.fn().mockImplementation(async (url: string, options?: RequestInit) => {
      if (url.endsWith('/admin/benchmarks') && options?.method === 'POST') {
        return { ok: true, json: async () => ({ id: 'run-1', status: 'queued' }) };
      }
      return { ok: true, json: async () => ({ items: [] }) };
    });
    vi.stubGlobal('fetch', fetchMock);
    render(<QueryClientProvider client={new QueryClient()}><App /></QueryClientProvider>);
    fireEvent.click(screen.getByRole('button', { name: 'Benchmarks' }));
    await screen.findByText('Runs use the built-in multilingual dataset.');
    fireEvent.click(screen.getByRole('button', { name: 'Run benchmark' }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith('/v1/admin/benchmarks',
      expect.objectContaining({ method: 'POST', body: expect.stringContaining('low_priority') })));
    vi.unstubAllGlobals();
  });

  it('checks, loads, and activates an immutable model candidate', async () => {
    sessionStorage.setItem('adminToken', 'admin-secret');
    const fetchMock = vi.fn().mockImplementation(async (url: string, options?: RequestInit) => {
      if (url.endsWith('/admin/models') && !options?.method) return { ok: true, json: async () => ({
        active_model: 'BAAI/bge-reranker-v2-m3', candidate: null, rollback_available: false,
        models: [{ name: 'BAAI/bge-reranker-v2-m3', revision: '953dc6f', status: 'ready',
          loaded: true, device_support: ['cpu'], max_length: 1024,
          estimated_memory_bytes: 2500000000, average_latency_ms: null }],
      }) };
      return { ok: true, json: async () => ({ valid: true, status: 'ready' }) };
    });
    vi.stubGlobal('fetch', fetchMock);
    render(<QueryClientProvider client={new QueryClient()}><App /></QueryClientProvider>);
    fireEvent.click(screen.getByRole('button', { name: 'Models' }));
    await screen.findByDisplayValue('BAAI/bge-reranker-v2-m3');
    expect(screen.getByText('Used by console requests now:').parentElement?.textContent)
      .toContain('BAAI/bge-reranker-v2-m3');
    expect(screen.getByText('● Used now · ready')).toBeTruthy();
    fireEvent.change(screen.getByLabelText('Immutable revision'), { target: { value: 'abcdef1' } });
    fireEvent.click(screen.getByRole('button', { name: 'Check' }));
    await screen.findByText('Candidate is valid.');
    fireEvent.click(screen.getByRole('button', { name: 'Load candidate' }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith('/v1/admin/models/load',
      expect.objectContaining({ method: 'POST', body: expect.stringContaining('abcdef1') })));
    expect((screen.getByRole('button', { name: 'Activate candidate' }) as HTMLButtonElement).disabled)
      .toBe(true);
    vi.unstubAllGlobals();
  });

  it('renders request metadata without private payload text', async () => {
    sessionStorage.setItem('adminToken', 'admin-secret');
    const fetchMock = vi.fn().mockImplementation(async (url: string) => {
      if (url.includes('/admin/requests')) return { ok: true, json: async () => ({
        total: 1, size: 20, items: [{ request_id: 'request-123456', correlation_id: 'corr-123456789',
          timestamp: 1, documents_count: 2, model: 'test-model', device: 'cpu', latency_ms: 8,
          cache_hits: 1, status: 'success', truncation_count: 0 }],
      }) };
      return { ok: true, json: async () => ({}) };
    });
    vi.stubGlobal('fetch', fetchMock);
    render(<QueryClientProvider client={new QueryClient()}><App /></QueryClientProvider>);
    fireEvent.click(screen.getByRole('button', { name: 'Requests' }));
    await screen.findByText('test-model');
    expect(screen.getByText('8ms')).toBeTruthy();
    expect(screen.queryByText('private document')).toBeNull();
    vi.unstubAllGlobals();
  });

  it('expands a dashboard minute into requests and their payload details', async () => {
    sessionStorage.setItem('adminToken', 'admin-secret');
    location.hash = 'Dashboard';
    const request = {
      request_id: 'request-dashboard', correlation_id: 'correlation-dashboard',
      timestamp: 620, documents_count: 1, model: 'test-model', device: 'cpu',
      latency_ms: 8, cache_hits: 0, status: 'success', truncation_count: 0,
    };
    const fetchMock = vi.fn().mockImplementation(async (url: string) => {
      if (url.endsWith('/admin/dashboard')) return { ok: true, json: async () => ({
        model: { name: 'test-model', revision: 'revision', ready: true, device: 'cpu',
          backend: { backend: 'legacy_cross_encoder' } },
        redis: { available: true },
        resources: { cpu_percent: 5, ram_percent: 47, uptime_seconds: 60 },
      }) };
      if (url.includes('/admin/metrics/timeseries')) return { ok: true, json: async () => ({
        points: [{ timestamp: 600, requests: 1, latency_p95_ms: 8, cache_hits: 0 }],
      }) };
      if (url.endsWith('/admin/requests/request-dashboard')) return { ok: true, json: async () => ({
        ...request,
        query: 'Incoming dashboard query',
        documents: [{ id: 'document-1', text: 'Incoming document text' }],
        results: [{ id: 'document-1', score: 2.4, normalized_score: 0.91, rank: 1,
          text: 'Outgoing ranked document', cache_hit: false }],
      }) };
      if (url.includes('/admin/requests?page=1&size=100')) return { ok: true, json: async () => ({
        total: 1, size: 100, items: [request],
      }) };
      return { ok: true, json: async () => ({}) };
    });
    vi.stubGlobal('fetch', fetchMock);
    render(<QueryClientProvider client={new QueryClient()}><App /></QueryClientProvider>);

    fireEvent.click(await screen.findByRole('button', { name: /1 requests/ }));
    await screen.findByText('test-model');
    fireEvent.click(screen.getByRole('button', {
      name: 'Show details for request request-dashboard',
    }));

    expect(await screen.findByText('Incoming dashboard query')).toBeTruthy();
    expect(screen.getByText('Incoming document text')).toBeTruthy();
    expect(screen.getByText('Outgoing ranked document')).toBeTruthy();
    vi.unstubAllGlobals();
  });

  it('submits multiple independent batch requests', async () => {
    sessionStorage.setItem('adminToken', 'admin-secret');
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => ({
      responses: [], total_pairs: 3, latency_ms: 1,
    }) });
    vi.stubGlobal('fetch', fetchMock);
    render(<QueryClientProvider client={new QueryClient()}><App /></QueryClientProvider>);
    fireEvent.click(screen.getByRole('button', { name: 'Batch Playground' }));
    fireEvent.click(screen.getByRole('button', { name: 'Add request' }));
    fireEvent.change(screen.getByLabelText('Batch query 2'), { target: { value: 'Python' } });
    fireEvent.change(screen.getByLabelText('Batch documents 2'), {
      target: { value: 'Python backend experience' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Run batch' }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith('/v1/admin/rerank/batch',
      expect.objectContaining({ method: 'POST', body: expect.stringContaining('Python') })));
    const call = fetchMock.mock.calls.find(([url]) => url === '/v1/admin/rerank/batch');
    expect(JSON.parse(call?.[1]?.body as string).requests).toHaveLength(2);
    vi.unstubAllGlobals();
  });

  it('reorders documents while preserving their IDs', async () => {
    sessionStorage.setItem('adminToken', 'admin-secret');
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => ({ results: [] }) });
    vi.stubGlobal('fetch', fetchMock);
    render(<QueryClientProvider client={new QueryClient()}><App /></QueryClientProvider>);
    fireEvent.click(screen.getByRole('button', { name: 'Rerank Playground' }));
    fireEvent.click(screen.getAllByRole('button', { name: 'Move down' })[0]);
    fireEvent.click(screen.getByRole('button', { name: 'Run rerank' }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith('/v1/admin/rerank',
      expect.objectContaining({ method: 'POST' })));
    const call = fetchMock.mock.calls.find(([url]) => url === '/v1/admin/rerank');
    const documents = JSON.parse(call?.[1]?.body as string).documents;
    expect(documents.map((document: { id: string }) => document.id)).toEqual(['docker', 'kubernetes']);
    vi.unstubAllGlobals();
  });
});
