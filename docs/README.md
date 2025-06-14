# Machinery Documentation

This directory contains the complete documentation for the Machinery project, built using [mdBook](https://rust-lang.github.io/mdBook/).

## Structure

```
docs/
├── book.toml           # mdBook configuration
├── src/                # Documentation source files
│   ├── SUMMARY.md      # Table of contents
│   ├── README.md       # Main introduction
│   ├── foundation/     # Project philosophy and architecture
│   ├── setup/          # Installation and setup guides
│   ├── config/         # Configuration reference
│   ├── infrastructure/ # Infrastructure documentation
│   ├── contributing/   # Contribution guidelines
│   └── status/         # Current implementation status
├── theme/              # Custom styling
└── book/               # Generated documentation (gitignored)
```

## Building Documentation

### Prerequisites

Install mdBook and plugins:

```bash
cargo install mdbook
cargo install mdbook-linkcheck
cargo install mdbook-toc
cargo install mdbook-mermaid
```

### Local Development

```bash
# Build documentation
mdbook build

# Serve documentation locally (http://localhost:3000)
mdbook serve

# Serve on specific port
mdbook serve --port 3001

# Watch for changes and rebuild automatically
mdbook watch
```

### From Project Root

The main Makefile includes documentation commands:

```bash
# Build and serve documentation
make docs

# Just build documentation
make docs-build

# Check documentation links
make docs-check
```

## Deployment

Documentation is automatically deployed to GitHub Pages when changes are pushed to the `main` branch. The deployment is handled by the `.github/workflows/docs.yml` workflow.

### Manual Deployment

If needed, you can manually deploy:

```bash
# Build documentation
mdbook build

# Deploy to gh-pages branch (requires gh-pages setup)
# This is typically handled by GitHub Actions
```

## Documentation Philosophy

This documentation follows the project's core principle: **only document what has been implemented**. 

### Content Guidelines

- ✅ **Document implemented features**: Configuration, infrastructure, tooling
- ✅ **Document setup procedures**: Installation, development environment
- ✅ **Document current status**: Clear distinction between implemented and planned
- ❌ **Don't document unimplemented features**: No API docs for non-existent APIs
- ❌ **Don't document planned features**: Keep future plans in roadmap only

### Updating Documentation

As features are implemented:

1. Update the relevant documentation pages
2. Update the [Current Status](src/status/current.md) page
3. Move items from "Not Yet Implemented" to "Implemented Features"
4. Add new documentation sections as needed

## Writing Guidelines

### Markdown Style

- Use consistent heading levels
- Include code examples with proper syntax highlighting
- Use callout boxes for important information
- Include cross-references between related pages

### Code Examples

```bash
# Always include comments in code examples
make dev-up

# Show expected output when helpful
# Expected: Starting machinery development environment...
```

### Status Indicators

Use visual indicators for implementation status:

- ✅ Implemented feature
- ⏳ In progress
- ❌ Not yet implemented
- ⚠️ Partially implemented

## Maintenance

### Regular Tasks

- [ ] Update status page as features are implemented
- [ ] Check for broken links using `mdbook-linkcheck`
- [ ] Update screenshots and examples
- [ ] Review and update configuration examples

### Link Checking

```bash
# Check all links in documentation
mdbook-linkcheck

# Or use the make command
make docs-check
```

## Contribution

When contributing to documentation:

1. Follow the existing structure and style
2. Only document implemented features
3. Test documentation locally before submitting
4. Update the table of contents if adding new pages
5. Check links and cross-references

## Troubleshooting

### Common Issues

#### mdBook Not Found
```bash
# Install mdBook
cargo install mdbook
```

#### Port Already in Use
```bash
# Use different port
mdbook serve --port 3001
```

#### Broken Links
```bash
# Check and fix broken links
mdbook-linkcheck
```

#### Styling Issues
- Check `theme/custom.css` for custom styles
- Verify mdBook version compatibility
- Clear browser cache if changes don't appear

## Advanced Configuration

The `book.toml` file contains advanced configuration options:

- **Search**: Full-text search is enabled
- **Editing**: Direct GitHub editing links
- **Themes**: Custom CSS theming
- **Plugins**: Additional mdBook plugins

For more information, see the [mdBook documentation](https://rust-lang.github.io/mdBook/). 