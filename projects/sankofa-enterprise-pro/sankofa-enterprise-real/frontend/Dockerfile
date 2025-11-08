# Sankofa Enterprise V12 - Frontend Dockerfile
# Multi-stage build for optimized production image

# Build stage
FROM node:22-alpine AS builder

WORKDIR /app

# Copy package files
COPY package*.json ./
COPY .npmrc ./

# Install dependencies with legacy peer deps to resolve React version conflicts
RUN npm ci --legacy-peer-deps

# Copy source code
COPY . .

# Build the application
RUN npm run build

# Production stage
FROM nginx:alpine

# Install curl for health checks
RUN apk add --no-cache curl

# Copy built application
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy custom nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Create non-root user and set permissions
RUN addgroup -g 1001 -S sankofa && \
    adduser -S sankofa -u 1001 && \
    chown -R sankofa:sankofa /usr/share/nginx/html && \
    chown -R sankofa:sankofa /var/cache/nginx && \
    chown -R sankofa:sankofa /var/log/nginx && \
    chown -R sankofa:sankofa /etc/nginx/conf.d && \
    touch /var/run/nginx.pid && \
    chown -R sankofa:sankofa /var/run/nginx.pid

# Switch to non-root user
USER sankofa

# Expose port
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:80 || exit 1

# Start nginx in foreground
CMD ["nginx", "-g", "daemon off;"]

