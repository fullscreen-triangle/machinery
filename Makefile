# Machinery - Continuous Individual Health Modeling Framework
# Makefile for building, testing, and running the system

.PHONY: help build test run clean setup check fmt clippy doc bench install dev prod

# Default target
help: ## Show this help message
	@echo "Machinery - Continuous Individual Health Modeling Framework"
	@echo ""
	@echo "Available targets:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Build targets
build: ## Build the entire project in debug mode
	cargo build

build-release: ## Build the project in release mode with optimizations
	cargo build --release

build-all: ## Build all workspace crates
	cargo build --workspace

# Test targets
test: ## Run all tests
	cargo test

test-unit: ## Run unit tests only
	cargo test --lib

test-integration: ## Run integration tests only
	cargo test --test '*'

test-doc: ## Run documentation tests
	cargo test --doc

test-coverage: ## Run tests with coverage report
	cargo tarpaulin --out Html --output-dir coverage/

# Code quality targets
check: ## Run cargo check on all targets
	cargo check --workspace --all-targets

fmt: ## Format code using rustfmt
	cargo fmt --all

fmt-check: ## Check if code is properly formatted
	cargo fmt --all -- --check

clippy: ## Run clippy linter
	cargo clippy --workspace --all-targets -- -D warnings

clippy-fix: ## Automatically fix clippy warnings where possible
	cargo clippy --workspace --all-targets --fix

# Documentation targets
doc: ## Generate documentation
	cargo doc --workspace --no-deps

doc-open: ## Generate and open documentation in browser
	cargo doc --workspace --no-deps --open

# Benchmarking targets
bench: ## Run benchmarks
	cargo bench

bench-baseline: ## Run benchmarks and save as baseline
	cargo bench -- --save-baseline main

bench-compare: ## Compare current benchmarks with baseline
	cargo bench -- --baseline main

# Installation and setup targets
install: ## Install the machinery binaries
	cargo install --path .

setup: ## Set up development environment
	@echo "Setting up Machinery development environment..."
	@mkdir -p data logs models seke_cache
	@cp machinery.toml machinery.local.toml 2>/dev/null || true
	@echo "Creating default directories..."
	@echo "Setup complete! Edit machinery.local.toml for local configuration."

setup-db: ## Set up databases (PostgreSQL and Redis)
	@echo "Setting up databases..."
	@docker-compose up -d postgres redis
	@sleep 5
	@cargo run --bin machinery-setup -- --init-db
	@echo "Database setup complete!"

# Running targets
run: ## Run the health AI orchestrator
	cargo run --bin machinery-orchestrator

run-setup: ## Run the setup utility
	cargo run --bin machinery-setup

run-cli: ## Run the CLI interface
	cargo run --bin machinery-cli

dev: ## Run in development mode with hot reloading
	@echo "Starting Machinery in development mode..."
	@RUST_LOG=debug cargo run --bin machinery-orchestrator -- --config machinery.local.toml

prod: ## Run in production mode
	@echo "Starting Machinery in production mode..."
	@cargo run --release --bin machinery-orchestrator -- --config machinery.toml

# Docker targets
docker-build: ## Build Docker image
	docker build -t machinery:latest .

docker-run: ## Run Machinery in Docker
	docker-compose up -d

docker-stop: ## Stop Docker containers
	docker-compose down

docker-logs: ## View Docker logs
	docker-compose logs -f machinery

# Database management targets
db-migrate: ## Run database migrations
	cargo run --bin machinery-setup -- --migrate

db-reset: ## Reset database (WARNING: destroys all data)
	@echo "WARNING: This will destroy all health data!"
	@read -p "Are you sure? (y/N): " confirm && [ "$$confirm" = "y" ]
	cargo run --bin machinery-setup -- --reset-db

db-backup: ## Create database backup
	@echo "Creating database backup..."
	@mkdir -p backups
	@pg_dump $(shell grep database_url machinery.toml | cut -d'"' -f2) > backups/machinery_backup_$(shell date +%Y%m%d_%H%M%S).sql
	@echo "Backup created in backups/ directory"

# Seke script management
seke-compile: ## Compile all Seke scripts
	cargo run --bin machinery-cli -- seke compile scripts/

seke-validate: ## Validate Seke scripts syntax
	cargo run --bin machinery-cli -- seke validate scripts/

seke-test: ## Test Seke scripts with sample data
	cargo run --bin machinery-cli -- seke test scripts/ --test-data test_data/

# Health data management
data-export: ## Export health data
	cargo run --bin machinery-cli -- data export --format json --output exports/

data-import: ## Import health data
	cargo run --bin machinery-cli -- data import --file $(FILE)

data-anonymize: ## Anonymize exported data
	cargo run --bin machinery-cli -- data anonymize --input $(INPUT) --output $(OUTPUT)

# Monitoring and maintenance
health-check: ## Check system health
	cargo run --bin machinery-cli -- health check

metrics: ## Display current metrics
	cargo run --bin machinery-cli -- metrics show

logs: ## View recent logs
	tail -f logs/machinery.log

logs-errors: ## View recent error logs
	grep ERROR logs/machinery.log | tail -20

# Security targets
security-audit: ## Run security audit
	cargo audit

security-update: ## Update dependencies for security
	cargo update

encrypt-config: ## Encrypt sensitive configuration
	cargo run --bin machinery-cli -- config encrypt machinery.local.toml

# Performance targets
profile: ## Profile the application
	cargo build --release
	perf record --call-graph=dwarf target/release/machinery-orchestrator
	perf report

flamegraph: ## Generate flamegraph
	cargo flamegraph --bin machinery-orchestrator

memory-check: ## Check for memory leaks with valgrind
	cargo build
	valgrind --tool=memcheck --leak-check=full target/debug/machinery-orchestrator

# Cleanup targets
clean: ## Clean build artifacts
	cargo clean

clean-all: ## Clean everything including data and logs
	cargo clean
	rm -rf data/ logs/ models/ seke_cache/ coverage/ target/

clean-data: ## Clean health data (WARNING: destroys all data)
	@echo "WARNING: This will destroy all health data!"
	@read -p "Are you sure? (y/N): " confirm && [ "$$confirm" = "y" ]
	rm -rf data/ models/ personal_baselines/

# Development workflow targets
pre-commit: fmt clippy test ## Run pre-commit checks
	@echo "Pre-commit checks passed!"

ci: check test clippy fmt-check doc ## Run CI pipeline locally
	@echo "CI pipeline completed successfully!"

release-check: ## Check if ready for release
	@echo "Checking release readiness..."
	@cargo check --release
	@cargo test --release
	@cargo clippy --release -- -D warnings
	@cargo fmt --check
	@cargo audit
	@echo "Release checks passed!"

# Quick development commands
quick-test: ## Quick test of core functionality
	cargo test --lib orchestrator modeling prediction

quick-build: ## Quick build of main binary only
	cargo build --bin machinery-orchestrator

watch: ## Watch for changes and rebuild
	cargo watch -x "build --bin machinery-orchestrator"

watch-test: ## Watch for changes and run tests
	cargo watch -x test

# Environment-specific targets
dev-setup: setup setup-db ## Complete development setup
	@echo "Development environment ready!"

prod-setup: ## Production environment setup
	@echo "Setting up production environment..."
	@mkdir -p /var/log/machinery /var/lib/machinery
	@cp machinery.toml /etc/machinery/
	@echo "Production setup complete!"

# Utility targets
version: ## Show version information
	@cargo run --bin machinery-cli -- --version

config-validate: ## Validate configuration file
	cargo run --bin machinery-cli -- config validate

device-list: ## List available health monitoring devices
	cargo run --bin machinery-cli -- devices list

device-test: ## Test device connections
	cargo run --bin machinery-cli -- devices test

# Example data targets
generate-sample-data: ## Generate sample health data for testing
	cargo run --bin machinery-cli -- data generate-sample --days 30 --output test_data/

load-sample-data: ## Load sample data into system
	cargo run --bin machinery-cli -- data import --file test_data/sample_health_data.json

# Backup and restore targets
backup: db-backup ## Create full system backup
	@echo "Creating full system backup..."
	@tar -czf backups/machinery_full_backup_$(shell date +%Y%m%d_%H%M%S).tar.gz data/ models/ machinery.toml
	@echo "Full backup created!"

restore: ## Restore from backup
	@echo "Restoring from backup: $(BACKUP_FILE)"
	@tar -xzf $(BACKUP_FILE)
	@echo "Restore complete!"

# Variables for common paths
CONFIG_FILE ?= machinery.toml
LOG_LEVEL ?= info
RUST_LOG ?= $(LOG_LEVEL)

# Export environment variables
export RUST_LOG
