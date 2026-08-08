import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { describe, expect, it, vi } from 'vitest';

import { App, Login } from './main';

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
});
