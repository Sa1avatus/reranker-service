import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './e2e',
  use: { baseURL: 'http://127.0.0.1:8400' },
  webServer: { command: 'vite --host 127.0.0.1 --port 8400', port: 8400, reuseExistingServer: true },
});
