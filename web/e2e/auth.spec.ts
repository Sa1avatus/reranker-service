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
