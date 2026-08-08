import { expect, test } from '@playwright/test';

test('admin can sign in and log out', async ({ page }) => {
  await page.route('**/v1/admin/dashboard', route => route.fulfill({ json: { model: { ready: true } } }));
  await page.goto('/');
  await page.getByLabel('Admin token').fill('test-admin-token');
  await page.getByRole('button', { name: 'Sign in' }).click();
  await expect(page.getByRole('heading', { name: 'Dashboard' })).toBeVisible();
  await page.getByRole('button', { name: 'Log out' }).click();
  await expect(page.getByRole('button', { name: 'Sign in' })).toBeVisible();
});

test('admin validates runtime configuration', async ({ page }) => {
  await page.route('**/v1/admin/dashboard', route => route.fulfill({ json: {} }));
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
