# Build stage
FROM node:20-alpine AS builder

WORKDIR /app

# Install pnpm
RUN corepack enable && corepack prepare pnpm@9.0.0 --activate

# Copy workspace files
COPY package.json pnpm-workspace.yaml pnpm-lock.yaml* ./
COPY packages/api/package.json ./packages/api/
COPY packages/crypto/package.json ./packages/crypto/

# Install dependencies
RUN pnpm install --frozen-lockfile

# Copy source code
COPY packages/api ./packages/api
COPY packages/crypto ./packages/crypto
COPY tsconfig.json ./

# Build
RUN pnpm --filter @voting/crypto build
RUN pnpm --filter @voting/api build

# Production stage
FROM node:20-alpine AS runner

WORKDIR /app

# Create non-root user
RUN addgroup --system --gid 1001 voting && \
    adduser --system --uid 1001 voting

# Install pnpm
RUN corepack enable && corepack prepare pnpm@9.0.0 --activate

# Copy built application
COPY --from=builder /app/package.json ./
COPY --from=builder /app/pnpm-workspace.yaml ./
COPY --from=builder /app/packages/api/dist ./packages/api/dist
COPY --from=builder /app/packages/api/package.json ./packages/api/
COPY --from=builder /app/packages/crypto/dist ./packages/crypto/dist
COPY --from=builder /app/packages/crypto/package.json ./packages/crypto/

# Install production dependencies only
RUN pnpm install --prod --frozen-lockfile

# Switch to non-root user
USER voting

# Environment variables
ENV NODE_ENV=production
ENV PORT=3000

EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://localhost:3000/health || exit 1

CMD ["node", "packages/api/dist/app.js"]
