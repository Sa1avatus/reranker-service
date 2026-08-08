FROM node:22.14.0-alpine3.21 AS build
WORKDIR /app
COPY package.json pnpm-lock.yaml pnpm-workspace.yaml tsconfig.json vite.config.ts index.html ./
COPY src ./src
RUN corepack enable && pnpm install --frozen-lockfile && pnpm run build
FROM nginx:1.27.4-alpine3.21
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 8400
