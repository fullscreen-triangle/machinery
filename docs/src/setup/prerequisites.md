# Prerequisites

Before setting up Machinery for development, ensure you have the following tools and dependencies installed.

## System Requirements

### Operating System
- **Linux**: Ubuntu 20.04+ or equivalent
- **macOS**: macOS 11.0+ (Big Sur or later)
- **Windows**: Windows 10+ with WSL2 recommended

### Hardware Requirements
- **RAM**: Minimum 8GB, recommended 16GB+
- **CPU**: Multi-core processor (4+ cores recommended)
- **Storage**: At least 10GB free space for development environment
- **Network**: Stable internet connection for dependency downloads

## Development Tools

### Rust Toolchain
```bash
# Install rustup (Rust installer and version management tool)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Verify installation
rustc --version
cargo --version
```

The project uses Rust 2021 edition. The exact toolchain version is specified in `rust-toolchain.toml`.

### Docker and Container Tools
```bash
# Docker (required for containerized development)
# Follow official Docker installation guide for your OS
docker --version
docker-compose --version

# Verify Docker is running
docker run hello-world
```

### Git
```bash
# Git for version control
git --version

# Configure Git (if not already done)
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

## Documentation Tools

### mdBook (for documentation)
```bash
# Install mdBook for building documentation
cargo install mdbook

# Verify installation
mdbook --version
```

### Additional mdBook Plugins
```bash
# Install useful plugins
cargo install mdbook-linkcheck
cargo install mdbook-toc
cargo install mdbook-mermaid
```

## Database Dependencies

### PostgreSQL
```bash
# Install PostgreSQL for development
# Ubuntu/Debian:
sudo apt-get install postgresql postgresql-contrib

# macOS with Homebrew:
brew install postgresql

# Start PostgreSQL service
sudo systemctl start postgresql  # Linux
brew services start postgresql   # macOS
```

### Redis
```bash
# Install Redis for caching and session management
# Ubuntu/Debian:
sudo apt-get install redis-server

# macOS with Homebrew:
brew install redis

# Start Redis service
sudo systemctl start redis-server  # Linux
brew services start redis          # macOS
```

## Optional Development Tools

### Code Editor/IDE
- **VS Code** with Rust-analyzer extension (recommended)
- **IntelliJ IDEA** with Rust plugin
- **Vim/Neovim** with appropriate Rust plugins

### Additional CLI Tools
```bash
# cargo-watch for automatic rebuilds during development
cargo install cargo-watch

# cargo-audit for security vulnerability scanning
cargo install cargo-audit

# cargo-outdated for dependency updates
cargo install cargo-outdated

# ripgrep for fast text searching
cargo install ripgrep
```

## Environment Verification

After installing all prerequisites, verify your environment:

```bash
# Clone the repository
git clone https://github.com/fullscreen-triangle/machinery.git
cd machinery

# Verify Rust toolchain
rustc --version
cargo --version

# Verify Docker
docker --version
docker-compose --version

# Verify database connections
psql --version
redis-cli --version

# Build documentation
cd docs
mdbook build
mdbook serve  # Serves docs at http://localhost:3000
```

## Troubleshooting

### Common Issues

#### Rust Installation Issues
- Ensure your shell's PATH includes `~/.cargo/bin`
- Restart your terminal after installing rustup
- Run `rustup update` to ensure latest stable version

#### Docker Permission Issues (Linux)
```bash
# Add your user to the docker group
sudo usermod -aG docker $USER
# Log out and back in for changes to take effect
```

#### Database Connection Issues
- Ensure PostgreSQL and Redis services are running
- Check default ports: PostgreSQL (5432), Redis (6379)
- Verify user permissions for database access

## Next Steps

Once all prerequisites are installed and verified, proceed to:
1. [Installation](installation.md) - Clone and set up the project
2. [Docker Environment](docker.md) - Configure containerized development
3. [Development Workflow](workflow.md) - Understand the development process 