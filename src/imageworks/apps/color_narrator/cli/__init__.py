"""Color-Narrator CLI module.

This module provides command line interfaces for VLM-guided color narration,
following the same patterns as the mono-checker CLI structure.

Available commands:
- narrate: Generate color descriptions using VLM inference
- validate: Validate existing color descriptions against data
"""

from .main import app

__version__ = "0.1.0"
__all__ = ["app"]
