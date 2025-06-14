# Machinery - Continuous Individual Health Modeling Framework
# Multi-stage Dockerfile for development and production

# Development stage with all tools
FROM rust:1.75-slim as development

# Install system dependencies for development
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    libpq-dev \
    libsqlite3-dev \
    build-essential \
    curl \
    git \
    postgresql-client \
    redis-tools \
    valgrind \
    perf-tools-unstable \
    && rm -rf /var/lib/apt/lists/*

# Install additional Rust tools for development
RUN cargo install cargo-watch cargo-audit cargo-tarpaulin cargo-flamegraph

# Set working directory
WORKDIR /workspace

# Copy workspace configuration
COPY rust-toolchain.toml ./
COPY Cargo.toml ./
COPY Cargo.lock ./

# Create workspace structure
RUN mkdir -p src crates/{orchestrator,seke-engine,data-collectors,prediction,modeling,validation,patterns}

# Copy source code
COPY . .

# Development command
CMD ["cargo", "build"]

# Build stage
FROM rust:1.75-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    libpq-dev \
    libsqlite3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create app user for security
RUN useradd -m -u 1001 machinery

# Set working directory
WORKDIR /app

# Copy workspace configuration first for better caching
COPY rust-toolchain.toml ./
COPY Cargo.toml ./
COPY Cargo.lock ./

# Create dummy source files to cache dependencies
RUN mkdir -p src/bin crates/{orchestrator,seke-engine,data-collectors,prediction,modeling,validation,patterns}/src
RUN echo "fn main() {}" > src/main.rs
RUN echo "fn main() {}" > src/bin/orchestrator.rs
RUN echo "fn main() {}" > src/bin/setup.rs
RUN echo "fn main() {}" > src/bin/cli.rs

# Create dummy Cargo.toml files for workspace crates
RUN for crate in orchestrator seke-engine data-collectors prediction modeling validation patterns; do \
    echo '[package]' > crates/$crate/Cargo.toml && \
    echo "name = \"machinery-$crate\"" >> crates/$crate/Cargo.toml && \
    echo 'version = "0.1.0"' >> crates/$crate/Cargo.toml && \
    echo 'edition = "2021"' >> crates/$crate/Cargo.toml && \
    echo '[dependencies]' >> crates/$crate/Cargo.toml && \
    echo 'pub fn dummy() {}' > crates/$crate/src/lib.rs; \
done

# Build dependencies (this layer will be cached)
RUN cargo build --release --bins
RUN rm -rf src crates

# Copy actual source code
COPY src ./src
COPY crates ./crates

# Build the actual application
RUN cargo build --release --bins

# Runtime stage
FROM debian:bookworm-slim as runtime

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd -m -u 1001 machinery

# Create necessary directories
RUN mkdir -p /app/{data,logs,models,scripts,seke_cache} && \
    chown -R machinery:machinery /app

# Copy binaries from builder stage
COPY --from=builder /app/target/release/machinery-orchestrator /usr/local/bin/
COPY --from=builder /app/target/release/machinery-setup /usr/local/bin/
COPY --from=builder /app/target/release/machinery-cli /usr/local/bin/

# Copy configuration files
COPY machinery.toml /app/
COPY scripts/ /app/scripts/

# Set permissions
RUN chmod +x /usr/local/bin/machinery-*
RUN chown -R machinery:machinery /app

# Switch to app user
USER machinery

# Set working directory
WORKDIR /app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose ports
EXPOSE 8080 9090

# Default command
CMD ["machinery-orchestrator", "--config", "/app/machinery.toml"]

# Production stage with additional optimizations
FROM runtime as production

# Copy production configuration
COPY machinery.production.toml /app/machinery.toml

# Set production environment variables
ENV RUST_LOG=info
ENV MACHINERY_ENV=production

# Use production command
CMD ["machinery-orchestrator", "--config", "/app/machinery.toml", "--production"]

# Minimal stage for CI/testing
FROM rust:1.75-slim as ci

# Install minimal dependencies for CI
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    libpq-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install CI tools
RUN cargo install cargo-audit cargo-tarpaulin

WORKDIR /workspace

# Copy source
COPY . .

# Default CI command
CMD ["cargo", "test", "--workspace"] 