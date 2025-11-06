# Proposal: Model Downloader Architecture Extraction

## 1. Context

Recent work added a VRAM Estimator that needs model architecture details
(layers, attention heads, KV precision, context limits, etc.). The current
registry does **not** store this metadata, so the GUI relies on generic defaults.
The model downloader is the logical place to parse and persist architecture data
immediately after files are fetched.

This proposal defines the data we should capture, how to extract it from common
formats (GGUF, AWQ/GPTQ, native HuggingFace safetensors), how to detect when it
is missing, and how to store it in the registry so downstream tools (VRAM
estimator, deployment planners, judge vision) can populate fields automatically.

## 2. Target Architecture Fields

| Field | Purpose | Needed by | Notes |
| --- | --- | --- | --- |
| `num_layers` | Decoder layers | VRAM estimator, judge scoring | Usually `num_hidden_layers` or `encoder_layers` in config. |
| `num_attention_heads` | Full attention heads | VRAM estimator (weights & KV) | `num_attention_heads`; GGUF metadata `llama.attention.head_count`. |
| `num_kv_heads` | KV heads (MHA vs MQA) | VRAM estimator | `num_key_value_heads`; GGUF `llama.attention.head_count_kv`. |
| `hidden_size` | Model width | VRAM estimator | `hidden_size`; GGUF `llama.embedding_length`. |
| `intermediate_size` | FFN width | Diagnostics | `intermediate_size` or `ffn_hidden_size`. |
| `rope_theta` / `rope_scaling` | Positional encoding info | VRAM planning, future features | Present in HF config; GGUF `general.rope_freq_base`. |
| `max_position_embeddings` / `context_length` | Max context supported | VRAM estimator inverse mode | Many configs include it; GGUF `llama.context_length`. |
| `vision_image_size`, `vision_patch`, `vision_dim` | Vision tower sizing | VRAM estimator (vision GiB) | For multimodal models (Qwen-VL) stored under `vision_config`. |
| `vocab_size` | Sanity for token budget | future use | Standard config field. |
| `attention_precision` / `kv_precision` | Actual runtime dtype | VRAM estimator (KV bytes) | Not always explicit; heuristics needed. |
| `quantization` | Already recorded but confirm | VRAM estimator (weight bytes) | Already tracked in registry. |

## 3. Format Analysis

### 3.1 HuggingFace Transformers Repos (safetensors / original weights)

- **Files**: `config.json`, optional `vision_config.json`,
  `generation_config.json`.
- **Available fields**: `num_hidden_layers`, `hidden_size`,
  `num_attention_heads`, `num_key_value_heads`, `max_position_embeddings`,
  `rope_theta`, `rope_scaling`, `vocab_size`.
- **Multimodal**: For Qwen-VL, `config.json` includes `vision_config`
  containing `image_size`, `patch_size`, `num_hidden_layers`.
- **Missing**: explicit KV dtype (defaults to model dtype). We may infer based on
  quantization type (fp16/bf16 → kv fp16, fp8 → kv fp8 if config includes TE
  hints); fallback to `hidden_size // num_attention_heads`.

### 3.2 GGUF (quantised weights)

- **Files**: `.gguf` holds metadata in header.
- **Extraction**: Use Python `gguf` reader (`gguf` package by Georgi Gerganov).
  Key fields:
  - `general.architecture`
  - `llama.context_length`
  - `llama.embedding_length`
  - `llama.block_count`
  - `llama.attention.head_count`
  - `llama.attention.head_count_kv`
  - `llama.attention.layer_norm_eps`
  - `llama.attention.classifier_dim`
  - `llama.rope.freq_base`, `llama.rope.freq_scale`
- **Quantization**: Provided (e.g., `quantization_type`).
- **Missing**: explicit intermediate size (some metadata includes `llama.feed_forward_length`, but not guaranteed); no vision tower info (these files are text-only).

### 3.3 GPTQ/AWQ repositories

- Typically include `config.json` identical to base model plus quantised weights
  (GGUF/GPTQ). If not present:
  - Attempt to read `.json` or `.yaml` metadata inside repo (`*.gptq_metadata`, `quantize_config.json`).
  - If configuration missing, fallback to fetching original base repo config (requires mapping to support repository, already supported by `--support-repo` flag).
- **Missing**: same as above (KV dtype, vision).

### 3.4 Safetensors with QLoRA/other adapters

- Usually no architecture change; rely on base config. Document fallback to
  support repo.

## 4. Proposed Pipeline

1. **Downloader hook**
   - After files download, run `ArchitectureExtractor.collect(model_dir)` that
     inspects the directory for known metadata (config, gguf).
   - Accept hints (`format_utils.detect_format_and_quant`) to determine whether
     to read GGUF vs config.

2. **ArchitectureExtractor**
   - Strategies (registered by format):
     - `HuggingFaceConfigStrategy`: parse `config.json` (and vision config if present).
     - `GGUFStrategy`: use `gguf.GGUFReader` to extract header metadata.
     - `SupportRepoStrategy`: if metadata missing, optionally fetch via support repo or original HF ID.
   - Normalise fields into `ArchitectureMetadata` dataclass:
     ```python
     @dataclass
     class ArchitectureMetadata:
         num_layers: Optional[int]
         num_attention_heads: Optional[int]
         num_kv_heads: Optional[int]
         hidden_size: Optional[int]
         intermediate_size: Optional[int]
         max_position_embeddings: Optional[int]
         rope_theta: Optional[float]
         rope_scaling: Optional[dict]
         context_length: Optional[int]
         vision: Optional[VisionMetadata]
         kv_precision: Optional[str]
         source: Literal["config.json", "gguf", "support_repo", "manual"]
     ```

3. **Registry Persistence**
   - Extend model registry schema: add `registry_entry["metadata"]["architecture"]`.
   - Example:
     ```json
     "metadata": {
       "architecture": {
         "num_layers": 36,
         "num_attention_heads": 16,
         "num_kv_heads": 8,
         "hidden_size": 6656,
         "max_position_embeddings": 8192,
         "context_length": 8192,
         "rope_theta": 100000.0,
         "vision": {
           "image_size": 448,
           "patch_size": 14,
           "num_layers": 24
         },
         "source": "config.json"
       }
     }
     ```
   - When data missing, omit field but record `source` and add `warnings` list for debugging.

4. **Registry Update Workflow**
   - Downloader writes metadata and saves registry entry (existing behaviour).
   - Provide CLI flag `--no-architecture` to skip extraction if necessary.
   - Build migration script to retro-populate architecture for installed models
     (iterate over download directories).

5. **Downstream Integration**
   - Update VRAM estimator page to load architecture metadata when a registry
     model is selected (populate fields, lock them by default).
   - Use architecture metadata to pre-select KV dtype (e.g., `config.get("rope_scaling", {}).get("factor")` etc.).
   - Future: personal tagger may adjust pipeline costs based on architecture.

## 5. Missing / Hard-to-Derive Details

| Detail | Availability | Notes / Plan |
| --- | --- | --- |
| Actual KV precision (runtime dtype) | Rarely explicit | Assume fp16/bf16 unless quantization indicates otherwise. For GGUF we can read `kv_precision` if metadata includes `llama.attention.key_precision`. Otherwise store `null` and let estimator default based on quantisation. |
| Vision tower VRAM overhead | Not encoded | Need heuristics (≈1 GiB default). Vision configs provide image size/patch but not actual runtime footprint—capture from loader logs when available. |
| Fragmentation factor | Environment-dependent | Remains profile-based (overhead JSON). Not derived from model file. |
| Custom positional embeddings | Some configs use `rope_scaling={"type": "yarn","factor":...}` | Record dictionary but estimator must interpret. |
| Adapter-specific architecture changes | LoRA/QLoRA don't alter base config | No extra data needed. |

## 6. Implementation Roadmap

1. **ArchitectureExtractor module**
   - Create `imageworks.tools.model_downloader.architecture` with strategies.
   - Unit tests reading sample config/gguf fixtures.
2. **Downloader integration**
   - After download, call extractor and merge metadata into registry entry before saving.
   - Extend `record_download` payload.
3. **Schema update**
   - Document new metadata fields; ensure JSON schema (if any) updated.
4. **Migration script**
   - CLI command `imageworks-download migrate-architecture` to reprocess existing downloads.
5. **Runtime reconciliation**
   - Instrument chat proxy / vLLM loader to log runtime metrics when a model initializes (KV dtype, effective max tokens, vision tower memory).
   - Persist to `logs/model_loader_metrics.jsonl`.
   - Provide `imageworks-download reconcile-architecture` command to merge runtime metrics into registry (`metadata.architecture.runtime`).
   - Add GUI button (Settings → Maintenance and VRAM Estimator) to trigger reconciliation after the first successful load.
6. **GUI enhancements & parameter propagation**
   - VRAM estimator: when registry model selected, prefill fields and annotate source (config vs runtime). Allow users to apply runtime overrides.
   - When estimator solves for an optimal context/batch combination, offer a “Apply to Model Settings” action that updates the registry entry’s vLLM `extra_args` (e.g. `--max-model-len`, `--max-num-seqs`, `--gpu-memory-utilization`) via the Models page backend.
   - Provide inline guidance differentiating vLLM `--max-model-len` (in tokens) versus Ollama `--num-ctx` (context window), and allow explicit mapping (user selects which flag to update).
7. **Documentation & tests**
   - Update model downloader reference/runbook.
   - Add tests verifying registry entry includes architecture after download.
   - Add coverage for reconciliation routine (mock metrics log → registry update) and parameter propagation helpers.

## 7. Risks & Mitigations

- **Incomplete metadata**: Some repos lack config; fallback to heuristics and emit warnings.
- **GGUF dependency**: Need to include `gguf` Python package; ensure license compatible.
- **Registry bloat**: Additional metadata increases JSON size marginally (<1 KB per entry).
- **Backward compatibility**: Ensure loaders ignore missing fields; default to behaviour when `architecture` absent.

## 8. Open Questions

1. Should we store `head_dim` explicitly or recompute (`hidden_size // heads`)?
2. Do we want to record encoder-only or decoder-only details differently (e.g.,
   for vision encoders)? For now treat everything as decoder-style.
3. Should VRAM estimator auto-select profile based on architecture (e.g., add
   `vision` boolean)? Likely yes once metadata present.

---

Deliverable: implement extraction pipeline as described and update GUI/CLI to
consume metadata. Most architectural detail can be recovered; the main missing
piece is reliable KV precision and vision tower VRAM, which will still rely on
profiles or manual overrides.
