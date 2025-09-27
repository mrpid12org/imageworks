# Test Color Narrator Directory

This directory contains test data and utilities for the color narrator system. **Large test files are excluded from git** to keep the repository lean.

## Directory Structure

```
test_color_narrator/
├── images/                      # Test images (not in git)
├── overlays/                    # Generated overlays (not in git)
├── production_images/           # Production test images (not in git)
├── *.jsonl                     # Test results (not in git)
├── *.log                       # Server logs (not in git)
├── mock_vllm_server.py         # Mock server for testing (in git)
└── README.md                   # This file (in git)
```

## Excluded from Git

The following files/directories are excluded via `.gitignore`:
- `images/` - Test image files
- `overlays/` - Generated visualization overlays
- `production_images/` - Production test image sets
- `*.jsonl` - Test result data files
- `*.log` - Server and test logs

## Setting Up Test Data

To run tests, you'll need to:

1. Add test images to `images/` directory
2. Run mono analysis to generate baseline results
3. Copy production images to `production_images/` if needed

## Mock vLLM Server

For testing without a real vLLM server:

```bash
cd test_color_narrator
python mock_vllm_server.py
```

This provides a lightweight mock API compatible with the color narrator system.
