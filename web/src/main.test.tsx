import { fireEvent, render, screen } from '@testing-library/react';
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
