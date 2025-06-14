# Installation

This guide will help you set up the Machinery project for development. Before proceeding, ensure you have completed all [Prerequisites](prerequisites.md).

## Quick Start

For a rapid setup using Docker (recommended for development):

```bash
# Clone the repository
git clone https://github.com/fullscreen-triangle/machinery.git
cd machinery

# Start the development environment
make dev-up

# Verify the setup
make health-check
```

## Manual Installation

### 1. Clone the Repository

```bash
git clone https://github.com/fullscreen-triangle/machinery.git
cd machinery
```

### 2. Verify Rust Toolchain

The project uses a specific Rust toolchain defined in `rust-toolchain.toml`:

```bash
# Check current Rust version
rustc --version

# Install/update to the required toolchain
rustup update
```

### 3. Install Dependencies

```bash
# Install Rust dependencies (this will download and compile dependencies)
cargo fetch

# Install additional development tools
cargo install cargo-watch cargo-audit cargo-outdated
```

### 4. Database Setup

#### PostgreSQL Setup
```bash
# Create development database
sudo -u postgres createdb machinery_dev

# Create test database
sudo -u postgres createdb machinery_test

# Verify connection
psql -h localhost -U postgres -d machinery_dev -c "SELECT version();"
```

#### Redis Setup
```bash
# Verify Redis is running
redis-cli ping
# Should return: PONG
```

### 5. Environment Configuration

Create environment-specific configuration files:

```bash
# Copy example configuration
cp machinery.toml.example machinery.toml

# Edit configuration for your environment
nano machinery.toml
```

### 6. Build Verification

Currently, the project is in infrastructure setup phase. Verify the configuration:

```bash
# Verify Cargo.toml is valid
cargo check

# Verify Docker setup
docker-compose config

# Build documentation
cd docs
mdbook build
mdbook serve  # View at http://localhost:3000
```

## Docker Installation (Recommended)

Using Docker provides a consistent development environment:

### 1. Using Docker Compose

```bash
# Clone and enter the project
git clone https://github.com/fullscreen-triangle/machinery.git
cd machinery

# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### 2. Using Makefile Commands

The project includes a comprehensive Makefile:

```bash
# Start development environment
make dev-up

# View all available commands
make help

# Run tests (when implemented)
make test

# Build for production
make build

# Clean up development environment
make clean
```

## Available Make Commands

| Command | Description |
|---------|-------------|
| `make dev-up` | Start development environment |
| `make dev-down` | Stop development environment |
| `make build` | Build the project |
| `make test` | Run all tests |
| `make clean` | Clean build artifacts |
| `make docs` | Build and serve documentation |
| `make lint` | Run code linting |
| `make format` | Format code |
| `make health-check` | Verify system health |

## Verification

After installation, verify everything is working:

### 1. Environment Check
```bash
# Check Rust installation
rustc --version
cargo --version

# Check Docker
docker --version
docker-compose --version

# Check databases
psql --version
redis-cli --version
```

### 2. Project Check
```bash
# Verify configuration
cargo check

# Verify Docker setup
docker-compose config

# Test database connections
make health-check
```

### 3. Documentation Check
```bash
# Build and serve documentation
cd docs
mdbook serve

# Open browser to http://localhost:3000
```

## Troubleshooting

### Common Issues

#### Cargo Check Fails
```bash
# Error: Cannot find workspace members
# Solution: This is expected as source code is not yet implemented
# The configuration files are ready for when implementation begins
```

#### Docker Issues
```bash
# Permission denied (Linux)
sudo usermod -aG docker $USER
# Log out and back in

# Port already in use
docker-compose down
sudo lsof -i :5432  # Check what's using PostgreSQL port
sudo lsof -i :6379  # Check what's using Redis port
```

#### Database Connection Issues
```bash
# PostgreSQL not running
sudo systemctl start postgresql  # Linux
brew services start postgresql   # macOS

# Redis not running
sudo systemctl start redis-server  # Linux
brew services start redis          # macOS

# Connection refused
# Check if services are bound to correct interfaces
netstat -tlnp | grep 5432  # PostgreSQL
netstat -tlnp | grep 6379  # Redis
```

## Development Workflow

Once installed, follow this workflow:

1. **Start Environment**: `make dev-up`
2. **Edit Configuration**: Modify `machinery.toml` as needed
3. **Verify Setup**: `make health-check`
4. **View Documentation**: `make docs`
5. **When Ready for Implementation**: Create source code in workspace crates

## Next Steps

After successful installation:

1. Read [Docker Environment](docker.md) for containerized development
2. Review [Development Workflow](workflow.md) for contribution guidelines
3. Check [Current Status](../status/current.md) to understand what's implemented
4. Explore [Configuration Reference](../config/project.md) for system setup

## Uninstallation

To remove the development environment:

```bash
# Stop and remove containers
make clean

# Remove Docker images (optional)
docker image prune

# Remove cloned repository
cd ..
rm -rf machinery
``` 