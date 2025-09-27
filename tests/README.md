# Test Directory Structure

This directory contains all tests organized by module with consistent naming and clear separation between unit tests and integration tests.

## Directory Structure

```
tests/
├── shared/                           # Shared test assets (NOT in git)
│   ├── images/                       # Real competition images for integration tests
│   ├── overlays/                     # Corresponding overlay files
│   └── sample_data/                  # Shared sample mono results, metadata, etc.
├── test_output/                      # Shared test output directory (NOT in git)
├── mono/                            # Mono checker tests
│   ├── unit/                        # Small unit tests (tracked in git)
│   │   ├── test_mono_core.py
│   │   ├── test_mono_*.py
│   │   └── fixtures/                # Small test images/data (tracked in git)
│   └── integration/                 # Integration tests using shared assets
├── color_narrator/                  # Color narrator tests
│   ├── unit/                        # Small unit tests (tracked in git)
│   │   ├── test_vlm.py
│   │   ├── test_hybrid_enhancer.py
│   │   ├── test_*.py
│   │   └── fixtures/                # Small test images/data (tracked in git)
│   ├── integration/                 # Integration tests using shared assets
│   └── experimental_results/        # Large test result files (NOT in git)
└── vision/                          # Vision library tests
    ├── unit/
    └── integration/
```

## Key Principles

### 1. **Separation by Module**
- Each major module has its own test directory
- Tests are isolated and can be run independently
- Clear ownership and responsibility

### 2. **Unit vs Integration Tests**
- **Unit tests** (`unit/`): Fast, small, tracked in git
  - Use small synthetic test images and mock data
  - Run frequently during development
  - Include fixtures directory for test assets
- **Integration tests** (`integration/`): Use real data from shared assets
  - Test end-to-end workflows
  - Use production-like images and data
  - May take longer to run

### 3. **Shared Assets (Not in Git)**
- `shared/`: Real competition images and production data
- `test_output/`: Shared output directory for test results
- Large files that don't need version control
- Prevents git bloat and reduces clone times

### 4. **Unit Test Fixtures (In Git)**
- Small synthetic or sample test files
- Essential for unit tests to work
- Tracked in git as they're small and necessary
- Located in each module's `unit/fixtures/` directory

## Usage Examples

### Running Tests by Module
```bash
# Run all mono tests
pytest tests/mono/

# Run only unit tests for color narrator
pytest tests/color_narrator/unit/

# Run specific test file
pytest tests/mono/unit/test_mono_core.py
```

### Using Shared Assets in Integration Tests
```python
# In integration tests, reference shared assets
SHARED_IMAGES = Path("tests/shared/images")
SHARED_DATA = Path("tests/shared/sample_data")
TEST_OUTPUT = Path("tests/test_output")
```

### Development Workflow
1. **Unit tests**: Run constantly during development - fast, reliable
2. **Integration tests**: Run before commits/releases - slower, uses real data
3. **Experimental results**: Archive large test output files for analysis

### ⚠️ **Important: Keep Project Root Clean**
**Always direct test outputs to `tests/test_output/` - never to project root!**

❌ **Don't do this:**
```bash
# Creates clutter in project root
enhance-mono -j data.jsonl -o results_test_001.json
analyze-regions --output manual_grid_test.jsonl
```

✅ **Do this instead:**
```bash
# Keep outputs organized in test directory
enhance-mono -j data.jsonl -o tests/test_output/results_001.json
analyze-regions --output tests/test_output/manual_grid_test.jsonl
```

**Why this matters:**
- Project root stays clean and professional
- Test outputs don't accidentally get committed to git
- Easier to find and clean up temporary files
- Clear separation between code and test artifacts

### CLI Testing Best Practices
```bash
# Use test output directory for all temporary files
--output tests/test_output/my_test.jsonl
--summary tests/test_output/my_summary.md
--results-json tests/test_output/analysis.json

# Use shared directory for reusable test assets
--images tests/shared/images/production_images
--mono-jsonl tests/shared/sample_data/production_sample.jsonl
```

## Migration Notes

This structure was created by reorganizing:
- `test_color_narrator/` → Distributed to appropriate module directories
- Production images → `tests/shared/images/`
- Experimental JSONL files → `tests/*/experimental_results/`
- Unit test fixtures → `tests/*/unit/fixtures/`

The new structure provides better organization, faster unit tests, and cleaner git history.
