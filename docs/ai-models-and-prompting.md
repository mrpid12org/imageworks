# AI Models and Prompting Guide

**Comprehensive guide to AI models, prompting strategies, and experimentation across the imageworks project**

---

## Table of Contents

1. [Overview](#overview)
2. [Current Model Implementations](#current-model-implementations)
3. [Model Experiments and Comparisons](#model-experiments-and-comparisons)
4. [Prompt Engineering](#prompt-engineering)
5. [Hardware Requirements](#hardware-requirements)
6. [Future Model Integration Plans](#future-model-integration-plans)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

---

## Overview

The imageworks project employs AI models across several components for computer vision and natural language processing tasks:

- **Vision-Language Models (VLM)**: For image analysis and description generation
- **Language Models (LLM)**: Planned for enhanced text generation and reasoning
- **Vision Models**: For specialized image processing tasks

This document consolidates all AI model usage, experiments, and lessons learned across the entire project.

---

## Current Model Implementations

### 1. Color-Narrator VLM System

**Purpose**: Analyze monochrome images for color contamination and generate professional descriptions.

#### Production Model: Qwen2.5-VL-7B-AWQ (LMDeploy)
- **Model Size**: ~7.3 GB quantised weights (two shards)
- **Total VRAM Usage**: ~13 GB on an RTX 4080 (16 GB) with eager mode; comfortably fits when vision batch size is 1
- **Inference Speed**: ~1.5–2.0 s per image (debug tests on 4080)
- **Quality**: Noticeably richer spatial grounding and cleaner prose than the 2 B baseline while keeping VRAM within consumer limits
- **Server**: LMDeploy TurboMind backend (`uv run lmdeploy serve api_server …`)
- **Notes**: Default LMDeploy port 24001; weights live at `$IMAGEWORKS_MODEL_ROOT/Qwen2.5-VL-7B-Instruct-AWQ`
- **Registry Roles**: `caption`, `description`, `keywords` (multi-purpose vision model) — see `configs/model_registry.json`.

#### Secondary Model: Qwen2-VL-2B-Instruct (vLLM)
- **Model Size**: 4.2 GB
- **Total VRAM Usage**: ~11 GB on RTX 4080
- **Inference Speed**: <1 s per image
- **Quality**: Still solid for contamination calls; kept for lightweight setups or as a regression check
- **Server**: vLLM OpenAI API (`scripts/start_vllm_server.py`)
- **Registry Roles**: `caption`, `description` (lightweight alternative)

#### Alternative Models Tested

**Qwen2-VL-7B-Instruct** ❌
- **Status**: Failed - CUDA OOM on RTX 4080 16GB
- **Model Size**: 13.6GB (too large)
- **Lesson**: Model size alone doesn't predict VRAM usage; vLLM overhead is ~2.6x model size

**Qwen2.5-VL-7B-Instruct-Q6K-GGUF** ❌
- **Status**: Not supported - GGUF quantisation targets llama.cpp/Ollama, not LMDeploy/vLLM
- **Lesson**: Stick to AWQ/GPTQ packages (or run GGUF via a separate runtime)

#### API Integration
```python
# VLM Client Configuration
VLMClient(
    base_url="http://localhost:24001/v1",
    model_name="Qwen2.5-VL-7B-AWQ",
    backend="lmdeploy",
    timeout=180,
)

# Request Structure (OpenAI-compatible)
{
    "model": "Qwen2.5-VL-7B-AWQ",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{overlay}"}}
            ]
        }
    ],
    "max_tokens": 300,
    "temperature": 0.1
}
```

### 2. Mono-Checker Analysis System

**Purpose**: Technical analysis of monochrome compliance using computer vision algorithms.

#### Current Implementation: Rule-Based CV
- **Technology**: OpenCV, PIL, NumPy, scikit-learn
- **Algorithms**: LAB color space analysis, clustering, statistical thresholds
- **Performance**: Fast processing, deterministic results
- **Integration**: Provides technical data that feeds into VLM prompts

#### Hybrid Enhancement: VLM + Mono Integration
- **Approach**: Combine mono-checker's technical accuracy with VLM's descriptive capabilities
- **Workflow**: Mono analysis → VLM enhancement → Human-readable summaries
- **Results**: 19 interesting cases from 244 total (filtering working effectively)

---

## Model Experiments and Comparisons

### VLM Model Performance Matrix

| Model | VRAM Required | Status | Quality Score | Speed | Use Case |
|-------|---------------|---------|---------------|-------|----------|
| Qwen2.5-VL-7B-AWQ (LMDeploy) | ≈13.5GB | ✅ Production | 9.5/10 (subjective) | Moderate (~1.5–2s) | Default narrator backend |
| Qwen2-VL-2B-Instruct | 10.88GB | ✅ Alternate | 9/10 | Fast (~1s) | Lightweight environments / regression checks |
| Qwen2-VL-7B-Instruct | >16GB | ❌ OOM | Unknown | Unknown | Would be better quality |
| Qwen2.5-VL-7B-Q6K-GGUF | N/A | ❌ Unsupported | Unknown | Unknown | Quantized fallback |

### Untested Sweet Spot Models
**Gap Analysis**: We only tested extremes (2B works, 7B fails). Potential candidates:
- Qwen2-VL-3B-Instruct: Likely fits in 16GB VRAM
- Qwen2-VL-4B-Instruct: May offer better quality than 2B
- Qwen2-VL-5B-Instruct: Upper limit for 16GB systems

**Recommendation**: Test intermediate models for quality/VRAM trade-offs.

### Hardware Scaling Requirements

#### Minimum (RTX 3060 12GB)
- Model: Qwen2-VL-2B-Instruct (fallback)
- VRAM Usage: 10.88GB (tight fit)
- Performance: Acceptable

#### Recommended (RTX 4080 16GB) ✅
- Model: Qwen2.5-VL-7B-AWQ (LMDeploy, eager mode)
- VRAM Usage: ~13.5GB / 16GB (≈84% utilisation)
- Performance: 1.5–2.0s per image
- Alternate: Swap to vLLM + Qwen2-VL-2B-Instruct if you need more headroom

#### High-End (RTX 6000 Pro 48GB)
- Models: All Qwen2-VL variants including 7B+
- Multiple concurrent model hosting possible
- Future TensorRT-LLM optimization potential

---

## Prompt Engineering
### Unified Registry & Role-Based Model Selection (Cross-App)
The project now uses a unified deterministic model registry (`configs/model_registry.json`) as the single source of truth for deployable models across Personal Tagger and (soon) Color Narrator.

Key points:
- Each model entry declares `roles` (e.g. `caption`, `keywords`, `description`, `narration`).
- Applications request functional roles instead of hard‑coding model names using `--use-registry` plus role flags (Personal Tagger: `--caption-role`, `--keyword-role`, `--description-role`).
- Central upgrades: change one registry entry + recompute/lock hash; no CLI/script edits.
- Drift detection: `verify <model> --lock` enforces reproducibility via aggregate artifact hashing.

Example (Personal Tagger):
```bash
uv run imageworks-personal-tagger run \
  -i ./images \
  --use-registry \
  --caption-role caption \
  --keyword-role keywords \
  --description-role description \
  --output-jsonl outputs/results/tagger.jsonl
```

Color Narrator will adopt the same pattern with a forthcoming `--use-registry` + `--vision-role` (see `color-narrator-reference.md`).

Further reading: `personal_tagger/model_registry.md` (Section 11) and `deterministic-model-serving.md` (Section 21.15).

#### When NOT to Use Explicit Model Names
Avoid hard-coding `model=...` in application CLI invocations when:
- You want centralized upgrades (most production + shared environments).
- Multiple roles map to the same underlying model (de-duplicates config churn).
- You rely on hash locking / reproducible deployments.

Still use explicit names temporarily when:
- Rapid local experiments with unregistered checkpoints.
- Benchmarking alternate weights before assigning roles.
- Debugging backend launch issues (use `--served-model-name` low-level flag).

### Evolution of Prompting Strategies

#### 1. Initial Simple Prompts (v1)
```python
# Basic color detection
prompt = f"""Describe where you see {dominant_color} color in this image."""
```
**Issues**: Generic responses, poor location specificity

#### 2. Context-Enhanced Prompts (v2)
```python
# Added technical context
prompt = f"""
Mono-checker analysis shows:
- Dominant color: {dominant_color}
- Verdict: {verdict}
Describe where you observe residual color.
"""
```
**Issues**: Better context but still hallucinated details ("zebra" references)

#### 3. Competition-Aware Prompts (v3)
```python
MONO_INTERPRETATION_TEMPLATE = """
You are a photography competition judge analyzing whether an image meets monochrome requirements.

TECHNICAL ANALYSIS DATA:
- Dominant Color: {dominant_color} (hue: {dominant_hue_deg:.1f}°)
- Colorfulness: {colorfulness:.2f}
- Chroma max: {chroma_max:.2f}

COMPETITION RULES:
- "Pass": True monochrome, no color contamination
- "Pass with query": Minor color issues that need review
- "Fail": Significant color contamination

Provide: VERDICT, TECHNICAL REASONING, VISUAL DESCRIPTION, PROFESSIONAL SUMMARY
"""
```
**Results**: More structured, competition-focused responses

#### 4. Location-Focused Prompts (v4 - Current Best)
```python
MONO_DESCRIPTION_ENHANCEMENT_TEMPLATE = """
You are analyzing a monochrome competition image that has been flagged by technical analysis for potential color issues.

CONTEXT:
- Image Title: "{title}" by {author}
- Dominant color detected: {dominant_color} (hue: {dominant_hue_deg:.1f}°)
- Technical data: colorfulness {colorfulness:.2f}, max chroma {chroma_max:.2f}

YOUR TASK: Look at this image and describe exactly WHERE you see the {dominant_color} color contamination.

Be very specific about locations - examples:
- "particularly around the zebra's mane and ears"
- "in the subject's hair on the left side"
- "on the dental clinic storefront signage"
- "appears in the background shadows"

COMPETITION RULING: Since this image has {dominant_color} color mixed with black/white/gray areas, it contains "shades of grey and another colour" and is NOT eligible for monochrome competition according to official rules.

Focus on WHERE the color appears, then explain why this disqualifies it from mono competition.
"""
```
**Results**: Specific location identification, competition rule compliance

#### 5. Region-Based Hallucination-Resistant Prompts (v5 - Current Best) 🆕
```python
REGION_BASED_COLOR_ANALYSIS_TEMPLATE = """
You are auditing a MONOCHROME competition photograph for residual colour.

You are given:
• Panel A: the original photograph.
• Panel B: an overlay marking WHERE colour appears (by hue direction).
• Panel C: an overlay showing HOW STRONG the colour is (brighter = stronger).
• A JSON list of REGIONS computed by technical analysis.

Truth constraints (very important):
1) Ground your description ONLY in the areas highlighted by Panels B/C and the supplied REGIONS.
2) Do NOT guess scene type, species, brands, or locations.
3) If you are uncertain, explicitly write "(uncertain)" at the end of the object/part phrase.
4) The tonal zone must be computed from mean_L (not guessed):
   - shadow if L* < 35; midtone if 35 ≤ L* < 70; highlight if L* ≥ 70.

Output format (strict):
• First, one bullet line per region, each ≤18 words
• Then a single JSON object with structured findings
"""
```

**Key Innovations**:
- **No Priming Examples**: Eliminates bias from "zebra" or other specific examples
- **Grounded Constraints**: Forces model to only describe what's in marked regions
- **Structured JSON Output**: Enables programmatic validation and confidence scoring
- **Uncertainty Handling**: Explicit "(uncertain)" escape hatch for ambiguous cases
- **Technical Integration**: Uses mono-checker's L* values for consistent tonal zones
- **Validation Pipeline**: Cross-checks VLM output against technical ground truth

**Implementation**: Available via `imageworks-color-narrator analyze-regions --demo`

### Prompt Template Categories

#### 1. **Technical Analysis Prompts**
- Focus: Numerical data interpretation
- Use Case: VLM verdict generation
- Structure: Technical context → Analysis → Verdict

#### 2. **Location Description Prompts**
- Focus: Spatial awareness and description
- Use Case: Human-readable color location identification
- Structure: Image context → Location examples → Specific task

#### 3. **Hybrid Enhancement Prompts**
- Focus: Combine technical accuracy with descriptive richness
- Use Case: Production workflow integration
- Structure: Mono verdict + VLM description → Enhanced output

#### 4. **Region-Based Analysis Prompts** 🆕
- Focus: Hallucination-resistant analysis with structured validation
- Use Case: Production-grade color contamination analysis
- Structure: Technical regions → Grounded analysis → Validated JSON output
- Key Features: Uncertainty handling, confidence scoring, cross-validation

### VLM Response Validation Pipeline

The region-based approach introduces comprehensive validation:

#### **Input Validation**
- Region data completeness (bbox, centroid, mean_L, hue_name)
- Technical consistency (L* values, hue angles, area percentages)
- Image format validation (base64 encoding, overlay alignment)

#### **Output Validation**
- JSON structure compliance (required fields, data types)
- Region index matching (VLM references valid input regions)
- Tonal zone verification (VLM output vs computed L* zones)
- Confidence range clamping (0.0-1.0 bounds enforcement)
- Cross-referencing hue names with technical analysis

#### **Quality Assurance**
- Low confidence detection (< 0.5 flagged for review)
- Uncertainty phrase detection ("(uncertain)" markers)
- Hallucination detection (descriptions not grounded in regions)
- Technical drift monitoring (VLM vs ground truth deviations)

### Prompt Optimization Principles

1. **Specific Examples**: Include concrete location examples to guide responses
2. **Competition Context**: Frame analysis in terms of photography competition rules
3. **Technical Grounding**: Provide numerical context to anchor descriptions
4. **Task Clarity**: Clearly define what type of response is expected
5. **Output Structure**: Specify format for consistent parsing
6. **Hallucination Prevention**: Use grounding constraints and validation pipelines
7. **Uncertainty Handling**: Provide explicit escape hatches for ambiguous cases

### Implementation: Region-Based Analysis System

#### **Command Line Interface**
```bash
# Test new approach with demo regions
uv run imageworks-color-narrator analyze-regions \
  --image photo.jpg \
  --demo \
  --debug \
  --output results.json

# Production usage (when mono-checker provides regions)
uv run imageworks-color-narrator analyze-regions \
  --image photo.jpg \
  --output results.json
```

#### **Sample Output Structure**
```json
{
  "file_name": "photo.jpg",
  "dominant_color": "yellow-green",
  "dominant_hue_deg": 88.0,
  "findings": [
    {
      "region_index": 0,
      "object_part": "subject's hair (uncertain)",
      "color_family": "yellow-green",
      "tonal_zone": "highlight",
      "location_phrase": "upper-left area",
      "confidence": 0.85
    }
  ],
  "validation_errors": []
}
```

#### **Integration with Existing Workflow**
- **Input**: Mono-checker JSONL + overlay images + region data
- **Processing**: Hallucination-resistant VLM analysis + validation
- **Output**: Structured JSON + human-readable summary + XMP metadata
- **Fallbacks**: Graceful degradation when VLM unavailable

#### **Quality Metrics**
- **Confidence Distribution**: Track high/medium/low confidence findings
- **Validation Error Rate**: Monitor technical inconsistencies
- **Uncertainty Rates**: Track "(uncertain)" usage patterns
- **Cross-Validation**: Compare VLM output with technical ground truth

---

## Hardware Requirements

### GPU Memory Analysis

**Why vLLM Uses More VRAM Than Model Size:**
- **Model Weights**: Base model file size (e.g., 4.2GB)
- **Activation Memory**: Temporary computation space (~60% of model size)
- **CUDA Graphs**: Pre-compiled execution optimizations (~100% of model size)
- **KV Cache**: Context/conversation memory
- **Overhead**: Framework and system overhead

**Formula**: `Total VRAM ≈ Model Size × 2.6`

### Recommended Configurations

#### Development Setup (RTX 4080 16GB)
```bash
# Optimal vLLM configuration
--gpu-memory-utilization 0.8    # Use 80% of available VRAM
--max-model-len 4096            # Sufficient for image+text context
--trust-remote-code             # Required for Qwen2-VL
--served-model-name "Qwen2-VL-2B-Instruct"
# (Note) For new pipelines prefer registry role-based resolution; direct --served-model-name
# remains for backend launch scripts & low-level tuning.
```

#### Production Setup (RTX 6000 Pro 48GB)
```bash
# Can run larger models
--gpu-memory-utilization 0.9    # Higher utilization safe with more VRAM
--max-model-len 8192            # Larger context window
--max-num-seqs 4                # Higher concurrency
```

### Server Deployment Best Practices

#### 1. Background Process Management
```bash
# vLLM helper (fallback Qwen2-VL-2B)
uv run python scripts/start_vllm_server.py

# LMDeploy helper (tQwen2.5-VL-7B eager mode)
uv run python scripts/start_lmdeploy_server.py --eager
```

#### 2. Health Monitoring
```python
# Automated health checks
def check_backend_health(base_url="http://localhost:8000/v1"):
    try:
        response = requests.get(f"{base_url}/models", timeout=10)
        return response.status_code == 200
    except Exception:
        return False
```

#### 3. Startup Time Management
- **Model Loading**: ~30 seconds
- **CUDA Graph Compilation**: ~30 seconds
- **Total Initialization**: ~60-90 seconds
- **Recommendation**: Wait 60+ seconds before first API call

---

## Future Model Integration Plans

### 1. Enhanced Color Description LLM

**Purpose**: Generate richer, more nuanced color analysis descriptions.

**Approach**:
- Use specialized LLM (non-vision) for text generation
- Input: Technical color statistics from mono-checker
- Output: Professional, detailed color analysis descriptions
- Integration: Complement VLM visual analysis with statistical interpretation

**Candidate Models**:
- Qwen2-7B-Instruct (text-only, more VRAM efficient)
- Llama 3.1-8B-Instruct (open source alternative)
- GPT-4o-mini via API (cloud option)

### 2. Personal Tagger VLM System

**Purpose**: General image tagging and metadata extraction for photography workflows.

**Approach**:
- Broader image analysis beyond monochrome detection
- Subject identification, composition analysis, technical quality assessment
- Integration with existing mono-checker and color-narrator systems

**Requirements**:
- Multi-domain image understanding
- Fast batch processing capabilities
- Structured metadata output

### 3. Multi-Modal Pipeline Architecture

**Vision**: Integrated AI pipeline with specialized models:

```
Image → Mono-Checker (CV) → Color-Narrator (VLM) → Description-LLM (LLM) → Personal-Tagger (VLM) → Final Metadata
```

**Benefits**:
- Each model optimized for specific tasks
- Modular architecture allowing independent upgrades
- Fallback options if any component fails
- Cost optimization through model size selection

---

## Best Practices

### Model Selection Criteria

1. **Task Alignment**: Match model capabilities to specific use case
2. **Hardware Constraints**: Ensure VRAM and compute requirements are feasible
3. **Quality Requirements**: Balance model size with output quality needs
4. **Latency Constraints**: Consider inference speed for interactive vs batch workflows
5. **Cost Considerations**: Local vs cloud deployment trade-offs

### Prompt Engineering Guidelines

1. **Iterative Development**: Test prompts with representative data samples
2. **Version Control**: Track prompt evolution and performance changes
3. **A/B Testing**: Compare prompt variants on same dataset
4. **Context Optimization**: Include just enough context without token bloat
5. **Output Validation**: Implement parsing and quality checks

### Production Deployment

1. **Resource Monitoring**: Track GPU utilization, memory usage, and inference latency
2. **Error Handling**: Implement robust fallback mechanisms for model failures
3. **Batch Optimization**: Group requests efficiently to maximize throughput
4. **Model Caching**: Pre-load models and maintain warm instances
5. **Logging**: Comprehensive logging for debugging and performance analysis

### Integration Patterns

1. **Hybrid Approaches**: Combine rule-based and AI-based methods
2. **Progressive Enhancement**: Use AI to enhance existing deterministic systems
3. **Graceful Degradation**: Fall back to simpler methods when AI components fail
4. **Modular Design**: Keep AI components loosely coupled for independent evolution

---

## Troubleshooting

### Common Issues and Solutions

#### 1. CUDA Out of Memory Errors
**Symptoms**: `torch.cuda.OutOfMemoryError` during model loading or inference

**Solutions**:
- Reduce `--gpu-memory-utilization` (try 0.6 or 0.7)
- Use smaller model variant (e.g., 2B instead of 7B)
- Decrease `--max-model-len` to reduce context memory
- Check for other GPU processes: `nvidia-smi`

#### 2. vLLM Server Startup Failures
**Symptoms**: Server exits immediately or fails to bind to port

**Solutions**:
- Verify model path exists and is readable
- Check port availability: `netstat -tlnp | grep 8000`
- Ensure `trust-remote-code` flag is set for custom models
- Check Python environment has vLLM properly installed

#### 3. API Connection Timeouts
**Symptoms**: Client requests timeout or fail to connect

**Solutions**:
- Increase client timeout values (default 120s may be insufficient)
- Wait longer after server startup (CUDA graph compilation takes time)
- Verify server health: `curl http://localhost:8000/v1/models`
- Check firewall and network configuration

#### 4. Poor Quality Responses
**Symptoms**: Generic, irrelevant, or inaccurate model outputs

**Solutions**:
- Review and refine prompt templates with specific examples
- Adjust temperature parameter (lower for more deterministic outputs)
- Add more context and constraints to prompts
- Consider using larger model variants if VRAM allows

#### 5. Slow Inference Performance
**Symptoms**: Requests take much longer than expected

**Solutions**:
- Reduce image resolution in requests
- Optimize batch sizes for your hardware
- Enable CUDA graph compilation (default in vLLM)
- Check for memory pressure causing swapping

### Debug Workflows

#### 1. Model Health Check
```bash
# Test server responsiveness
curl -s http://localhost:8000/v1/models | python3 -m json.tool

# Expected response structure
{
  "object": "list",
  "data": [
    {
      "id": "Qwen2-VL-2B-Instruct",
      "object": "model"
    }
  ]
}
```

#### 2. End-to-End Testing
```bash
# Run color-narrator with debug output
uv run imageworks-color-narrator narrate \
  --images test_color_narrator/images \
  --overlays test_color_narrator/overlays \
  --mono-jsonl test_color_narrator/test_mono_results.jsonl \
  --debug \
  --dry-run
```

#### 3. Performance Profiling
```python
import time
import psutil
import GPUtil

def profile_inference():
    start_time = time.time()
    gpu_before = GPUtil.getGPUs()[0]

    # Make VLM request
    response = vlm_client.infer_single(request)

    end_time = time.time()
    gpu_after = GPUtil.getGPUs()[0]

    print(f"Inference time: {end_time - start_time:.2f}s")
    print(f"GPU memory: {gpu_after.memoryUsed - gpu_before.memoryUsed}MB")
```

---

## Conclusion

The imageworks AI model ecosystem is designed for modularity, scalability, and production reliability. The current VLM implementation provides a solid foundation, and the documented experiments and lessons learned will guide future expansions.

Key takeaways:
- **Hardware-aware model selection** is critical for practical deployment
- **Prompt engineering** requires iterative refinement with domain-specific examples
- **Hybrid approaches** combining AI and traditional methods often outperform pure AI solutions
- **Production deployment** requires careful attention to resource management and error handling

This guide will be updated as new models are integrated and additional experiments are conducted across the imageworks project.
