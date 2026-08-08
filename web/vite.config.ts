import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
export default defineConfig({plugins:[react()],server:{proxy:{'/v1':'http://localhost:8200','/health':'http://localhost:8200'}},test:{environment:'jsdom',setupFiles:['./src/test-setup.ts'],include:['src/**/*.test.tsx']}});
