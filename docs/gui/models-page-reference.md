# Models Page Comprehensive Implementation - Quick Reference

## Overview
The Models page (`2_🎯_Models.py`) has been completely reorganized to expose all `model_downloader` CLI commands in the GUI with full feature parity.

## Structure (5 Tabs)

### 1. 📚 Browse & Manage
**Purpose:** Model browsing with role and configuration editing

**Features:**
- Registry statistics widget (expandable)
- Searchable/filterable model table (backend, format, quantization)
- Selected model details with size calculation
- Role assignment multiselect (8 roles)
- Role priority editor (0-100 per role)
- Backend configuration display
- Extra arguments editor (one per line)
- Common arguments helper (vLLM/LMDeploy/Ollama examples)
- Save/Reset controls

**CLI Equivalents:**
- Displays data from: `imageworks-download list --details`
- Saves to: unified registry via `save_registry()`

---

### 2. 📥 Download & Import
**Purpose:** Acquire new models from multiple sources

#### Sub-tab: 🌐 Download from HuggingFace
**Features:**
- Model identifier input (owner/repo or URL)
- Optional branch specification
- Format preferences (gguf, awq, gptq, safetensors)
- Location selection (linux_wsl, windows_lmstudio, custom)
- Include optional files checkbox
- Force re-download checkbox
- Non-interactive mode toggle
- Command preview expander
- Progress indicator with output display

**CLI Equivalent:**
```bash
uv run imageworks-download download <model> --format <fmt> --location <loc> [OPTIONS]
```

#### Sub-tab: 📁 Scan Existing
**Features:**
- Base directory input
- Dry-run toggle (default: ON)
- Update existing entries checkbox
- Include testing models checkbox
- Fallback format selector
- Command preview
- Progress indicator

**CLI Equivalent:**
```bash
uv run imageworks-download scan --base <dir> [--dry-run] [OPTIONS]
```

#### Sub-tab: 🦙 Import Ollama
**Features:**
- Dry-run toggle (default: ON)
- Deprecate placeholders checkbox
- Backend name input
- Location selector
- Command preview
- Progress indicator

**CLI Equivalent:**
```bash
uv run python scripts/import_ollama_models.py [--dry-run] [OPTIONS]
```

---

### 3. 🔧 Registry Maintenance
**Purpose:** Maintain registry consistency and cleanliness

⚠️ **Warning:** All operations show dry-run preview by default

#### Sub-tab: 🔄 Normalize
**Features:**
- Dry-run toggle (default: ON)
- Rebuild dynamic fields checkbox
- Prune missing entries checkbox
- Create backup toggle (default: ON)
- Command preview
- Preview/Apply button (label changes with dry-run)

**CLI Equivalent:**
```bash
uv run imageworks-download normalize-formats [--dry-run|--apply] [OPTIONS]
```

#### Sub-tab: 🗑️ Purge
**Operations:**
1. **purge-deprecated:** Remove deprecated entries (with placeholders-only option)
2. **purge-logical-only:** Remove entries without download_path (with curated option)
3. **purge-hf:** Remove HF entries from weights root (with backend filter)
4. **reset-discovered:** Reset discovered layer for backend

**CLI Equivalents:**
```bash
uv run imageworks-download purge-deprecated [--placeholders-only] [--dry-run]
uv run imageworks-download purge-logical-only [--include-curated] [--dry-run|--apply]
uv run imageworks-download purge-hf --weights-root <path> [--backend <name>] [--dry-run]
uv run imageworks-download reset-discovered --backend <name> [--dry-run]
```

#### Sub-tab: 🔨 Cleanup
**Operations:**
1. **prune-duplicates:** Remove duplicate variants keeping richest metadata
2. **restore-ollama:** Restore Ollama entries from backup
3. **backfill-ollama-paths:** Backfill synthetic paths for legacy entries

**CLI Equivalents:**
```bash
uv run imageworks-download prune-duplicates [--backend <name>] [--dry-run]
uv run imageworks-download restore-ollama [--backup <file>] [--include-deprecated] [--dry-run]
uv run imageworks-download backfill-ollama-paths [--dry-run]
```

---

### 4. 🔌 Backends
**Purpose:** Monitor backend status and resources

**Features:**
- Backend status monitor (reuses existing component)
- System resources display
- Manual start commands reference (expandable)

**Note:** Start/stop controls planned for future phase

---

### 5. ⚙️ Advanced Operations
**Purpose:** Dangerous operations requiring explicit confirmation

⚠️ **DANGER ZONE:** Can delete files permanently

#### Sub-tab: 🗑️ Remove Models
**Features:**
- Model variant selector (dropdown)
- Selected model details display (name, backend, format, quant, path, exists status)
- Removal mode radio buttons:
  - Metadata only (keep files & logical entry)
  - Files only (keep logical entry)
  - Purge entirely (delete entry + files)
- Confirmation checkbox (different text for destructive operations)
- Command preview
- Remove button (disabled until confirmed)
- Auto-refresh after removal

**CLI Equivalent:**
```bash
uv run imageworks-download remove <variant> [--delete-files] [--purge] --force
```

#### Sub-tab: ✅ Verify
**Features:**
- Verify all toggle (default: ON)
- Specific model selector (when verify-all is OFF)
- Auto-fix missing checkbox
- Command preview
- Verify button with progress

**CLI Equivalent:**
```bash
uv run imageworks-download verify [<variant>] [--fix-missing]
```

#### Sub-tab: 📊 Profiles
**Features:**
- Display current pyproject.toml configuration
- JSON viewer for tool.imageworks settings

**Note:** Visual profile editor planned for future phase

---

## Safety Features

### Default Behaviors
- ✅ **Dry-run by default:** All destructive operations default to `--dry-run`
- ✅ **Confirmation required:** File deletion requires explicit checkbox confirmation
- ✅ **Command preview:** Every operation shows exact CLI command before execution
- ✅ **Backup by default:** Registry modifications create timestamped backups

### User Feedback
- ✅ **Progress indicators:** `st.spinner()` for all long operations
- ✅ **Success/error messages:** Clear feedback with ✅/❌ icons
- ✅ **Output display:** Expandable output for detailed results
- ✅ **Error display:** stderr shown in error messages

### Helper Functions
```python
def run_command_with_progress(command, description, show_output=True, timeout=None)
    # Runs command with spinner, shows success/error, displays output

def confirm_destructive_operation(operation_name, details)
    # Shows warning + details + confirmation checkbox
    # Returns: bool (confirmed or not)
```

---

## Testing

### Unit Tests
```bash
# Test CLI command construction
uv run python tests/gui/test_models_page_commands.py
# Expected: 6 passed, 0 failed
```

### Import Test
```bash
# Verify page imports without errors
uv run python -c "import importlib.util; spec = importlib.util.spec_from_file_location('models_page', 'src/imageworks/gui/pages/2_🎯_Models.py'); module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); print('✅ Import successful')"
```

### Integration Test
1. Start GUI: `uv run streamlit run src/imageworks/gui/app.py --server.port=8501`
2. Navigate to Models page
3. Test each tab:
   - Browse & Manage: Select model, edit roles, save
   - Download & Import: Preview command for each sub-tab
   - Registry Maintenance: Run `normalize-formats --dry-run`
   - Backends: Check status display
   - Advanced: Preview remove command (DON'T actually remove unless testing)

---

## File Structure

```
src/imageworks/gui/pages/
├── 2_🎯_Models.py              # New comprehensive implementation (866 lines)
├── 2_🎯_Models.py.backup       # Original backup
└── 2_🎯_Models.py.old          # Previous version

tests/gui/
└── test_models_page_commands.py  # CLI command construction tests

scripts/
├── generate_comprehensive_models_page.py  # Generator script
└── generate_models_page.py               # Helper script
```

---

## Development Notes

### Growth Statistics
- Original: 672 lines (4 tabs)
- New: 866 lines (5 tabs)
- Growth: +194 lines (+29%)

### Function Count
1. `run_command_with_progress()` - Progress wrapper
2. `confirm_destructive_operation()` - Confirmation dialog
3. `render_browse_manage_tab()` - Tab 1
4. `render_download_import_tab()` - Tab 2 (3 sub-tabs)
5. `render_registry_maintenance_tab()` - Tab 3 (3 sub-tabs)
6. `render_backends_tab()` - Tab 4
7. `render_advanced_tab()` - Tab 5 (3 sub-tabs)
8. `main()` - Entry point

### CLI Commands Mapped
**Download/Import (3):**
- `imageworks-download download`
- `imageworks-download scan`
- `import_ollama_models.py`

**Maintenance (7):**
- `imageworks-download normalize-formats`
- `imageworks-download purge-deprecated`
- `imageworks-download purge-logical-only`
- `imageworks-download purge-hf`
- `imageworks-download reset-discovered`
- `imageworks-download prune-duplicates`
- `imageworks-download restore-ollama`
- `imageworks-download backfill-ollama-paths`

**Advanced (2):**
- `imageworks-download remove`
- `imageworks-download verify`

**Total: 12+ CLI commands fully exposed**

---

## Future Enhancements

1. **Download Progress:** Real-time progress bars for aria2c downloads
2. **Backend Start/Stop:** Integrated controls for vLLM/LMDeploy/Ollama
3. **Profile Editor:** Visual editor for pyproject.toml role assignments
4. **Batch Operations:** Multi-select for bulk remove/verify
5. **Stats Dashboard:** Separate tab with detailed charts
6. **Model Preview:** Quick model info cards with thumbnails
7. **Search/Filter:** Advanced filtering across all tabs
8. **History Log:** Track all operations with undo capability

---

## Troubleshooting

### GUI Won't Start
```bash
# Check if already running
ps aux | grep streamlit

# Kill existing process
pkill -f streamlit

# Restart
uv run streamlit run src/imageworks/gui/app.py --server.port=8501
```

### Import Errors
```bash
# Verify no syntax errors
uv run python -m py_compile src/imageworks/gui/pages/2_🎯_Models.py

# Check linter
uv run ruff check src/imageworks/gui/pages/2_🎯_Models.py
```

### Command Not Working
1. Check command preview in expander
2. Copy command and test in terminal manually
3. Verify CLI tool is installed: `uv run imageworks-download --help`
4. Check timeout settings (default 60-3600s depending on operation)

---

## Related Documentation

- **Model Downloader CLI:** `docs/reference/model-downloader.md`
- **Registry Architecture:** `docs/architecture/deterministic-model-serving.md`
- **GUI Configuration:** `src/imageworks/gui/config.py`
- **Process Runner:** `src/imageworks/gui/components/process_runner.py`
