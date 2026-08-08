import { expect, test } from '@playwright/test';

test('admin can sign in and log out', async ({ page }) => {
  await page.route('**/v1/admin/dashboard', route => route.fulfill({ json: {
    model: { ready: true, name: 'test', revision: 'abc', device: 'cpu' },
    redis: { available: true }, resources: { cpu_percent: 1, ram_percent: 2, uptime_seconds: 3 },
  } }));
  await page.route('**/v1/admin/metrics/timeseries**', route => route.fulfill({ json: { points: [] } }));
  await page.goto('/');
  await page.getByLabel('Admin token').fill('test-admin-token');
  await page.getByRole('button', { name: 'Sign in' }).click();
  await expect(page.getByRole('heading', { name: 'Dashboard' })).toBeVisible();
  await page.getByRole('button', { name: 'Log out' }).click();
  await expect(page.getByRole('button', { name: 'Sign in' })).toBeVisible();
});

test('admin validates runtime configuration', async ({ page }) => {
  await page.route('**/v1/admin/dashboard', route => route.fulfill({ json: {
    model: { ready: true }, redis: { available: true },
    resources: { cpu_percent: 1, ram_percent: 2, uptime_seconds: 3 },
  } }));
  await page.route('**/v1/admin/metrics/timeseries**', route => route.fulfill({ json: { points: [] } }));
  await page.route('**/v1/admin/runtime', route => route.fulfill({ json: {
    batch_size: 16, max_concurrency: 2, max_length: 1024, dynamic_batching: true,
    batch_window_ms: 10, max_batch_pairs: 128, request_timeout_seconds: 15,
    default_top_n: 10, queue_depth: 0,
  } }));
  await page.route('**/v1/admin/runtime/validate', route => route.fulfill({
    json: { valid: true, memory_warning: false, restart_required: false },
  }));
  await page.goto('/');
  await page.getByLabel('Admin token').fill('test-admin-token');
  await page.getByRole('button', { name: 'Sign in' }).click();
  await page.getByRole('button', { name: 'Runtime' }).click();
  await page.getByRole('button', { name: 'Validate configuration' }).click();
  await expect(page.getByText('Configuration is valid.')).toBeVisible();
});

test('admin starts a low-priority benchmark', async ({ page }) => {
  await page.route('**/v1/admin/dashboard', route => route.fulfill({ json: {
    model: { ready: true }, redis: { available: true },
    resources: { cpu_percent: 1, ram_percent: 2, uptime_seconds: 3 },
  } }));
  await page.route('**/v1/admin/metrics/timeseries**', route => route.fulfill({ json: { points: [] } }));
  await page.route('**/v1/admin/benchmarks', async route => {
    if (route.request().method() === 'POST') await route.fulfill({ json: { id: 'run-1', status: 'queued' } });
    else await route.fulfill({ json: { items: [] } });
  });
  await page.goto('/');
  await page.getByLabel('Admin token').fill('test-admin-token');
  await page.getByRole('button', { name: 'Sign in' }).click();
  await page.getByRole('button', { name: 'Benchmarks' }).click();
  const requestPromise = page.waitForRequest(request =>
    request.url().endsWith('/v1/admin/benchmarks') && request.method() === 'POST');
  await page.getByRole('button', { name: 'Run benchmark' }).click();
  const request = await requestPromise;
  expect(request.postDataJSON()).toMatchObject({ mode: 'low_priority', multilingual: true });
});

test('admin views paginated technical request records', async ({ page }) => {
  await page.route('**/v1/admin/dashboard', route => route.fulfill({ json: {
    model: { ready: true }, redis: { available: true },
    resources: { cpu_percent: 1, ram_percent: 2, uptime_seconds: 3 },
  } }));
  await page.route('**/v1/admin/metrics/timeseries**', route => route.fulfill({ json: { points: [] } }));
  await page.route('**/v1/admin/requests**', route => route.fulfill({ json: {
    total: 1, size: 20, items: [{ request_id: 'request-123456', correlation_id: 'corr-123456789',
      timestamp: 1, documents_count: 2, model: 'test-model', device: 'cpu', latency_ms: 8,
      cache_hits: 1, status: 'success', truncation_count: 0 }],
  } }));
  await page.goto('/');
  await page.getByLabel('Admin token').fill('test-admin-token');
  await page.getByRole('button', { name: 'Sign in' }).click();
  await page.getByRole('button', { name: 'Requests' }).click();
  await expect(page.getByText('test-model')).toBeVisible();
  await expect(page.getByText('1 retained technical records')).toBeVisible();
});

test('admin submits multiple requests through batch playground', async ({ page }) => {
  await page.route('**/v1/admin/dashboard', route => route.fulfill({ json: {
    model: { ready: true }, redis: { available: true },
    resources: { cpu_percent: 1, ram_percent: 2, uptime_seconds: 3 },
  } }));
  await page.route('**/v1/admin/metrics/timeseries**', route => route.fulfill({ json: { points: [] } }));
  await page.route('**/v1/admin/rerank/batch', route => route.fulfill({
    json: { responses: [], total_pairs: 3, latency_ms: 1 },
  }));
  await page.goto('/');
  await page.getByLabel('Admin token').fill('test-admin-token');
  await page.getByRole('button', { name: 'Sign in' }).click();
  await page.getByRole('button', { name: 'Batch Playground' }).click();
  await page.getByRole('button', { name: 'Add request' }).click();
  await page.getByLabel('Batch query 2').fill('Python experience');
  await page.getByLabel('Batch documents 2').fill('Python backend experience');
  const requestPromise = page.waitForRequest('**/v1/admin/rerank/batch');
  await page.getByRole('button', { name: 'Run batch' }).click();
  const payload = (await requestPromise).postDataJSON();
  expect(payload.requests).toHaveLength(2);
  await expect(page.getByRole('button', { name: 'Export JSON' })).toBeEnabled();
});

test('admin reorders documents without changing IDs', async ({ page }) => {
  await page.route('**/v1/admin/dashboard', route => route.fulfill({ json: {
    model: { ready: true }, redis: { available: true },
    resources: { cpu_percent: 1, ram_percent: 2, uptime_seconds: 3 },
  } }));
  await page.route('**/v1/admin/metrics/timeseries**', route => route.fulfill({ json: { points: [] } }));
  await page.route('**/v1/admin/rerank', route => route.fulfill({ json: { results: [] } }));
  await page.goto('/');
  await page.getByLabel('Admin token').fill('test-admin-token');
  await page.getByRole('button', { name: 'Sign in' }).click();
  await page.getByRole('button', { name: 'Rerank Playground' }).click();
  await page.getByRole('button', { name: 'Move down' }).first().click();
  const requestPromise = page.waitForRequest('**/v1/admin/rerank');
  await page.getByRole('button', { name: 'Run rerank' }).click();
  const payload = (await requestPromise).postDataJSON();
  expect(payload.documents.map((document: { id: string }) => document.id)).toEqual([
    'docker', 'kubernetes',
  ]);
});
