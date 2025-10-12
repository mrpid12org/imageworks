# 📘 Project Handover — ImageWorks AI Orchestration Stack  
*(LangFlow engine + optional Dify app layer)*  

**Prepared for:** ImageWorks project (Mono / Narrate / Similarity modules)  
**Audience:** developer or ML engineer assisting with implementation  
**Version:** 1.0 — October 2025  

---

## 1. Project Context

The ImageWorks repository (`mrpid12org/imageworks`) contains several Python modules for photographic image analysis:

- **Mono** — detect residual colour in competition monochrome images and output a short, judge-friendly comment.  
- **Narrate** — describe where colour appears in an image (“muted teal in the lower third”).  
- **Similarity** — determine whether two photographs are duplicates or variants (crop, tone, colour, etc.).  

The current codebase works but is **monolithic** and mixes logic, prompting, and CLI behaviour.  
Goal: refactor it into clear, modular, testable packages that can run in **LangFlow** (visual orchestration) and optionally surface in **Dify** (user-facing UI + evaluation).

---

## 2. Architecture Overview

### 2.1 System diagram

```
[process_photos.py / CLI tools]
        │     (HTTP POST/GET)
        ▼
+---------------------------+
|        LangFlow API       |
| (self-hosted, MIT)        |
|  - Custom Python nodes    |
|  - Runs Mono/Narrate/Sim  |
|  - Serves REST endpoints  |
+------------+--------------+
             │
             │ (HTTP Tools)
             ▼
+---------------------------+
|           Dify            |
| (self-hosted Community)   |
|  - UI & auth              |
|  - Workflows as Tools     |
|  - Iteration & retries    |
|  - Dataset evaluation     |
+---------------------------+
```

- **LangFlow** provides the **engine layer** — wraps existing algorithms, runs locally, exposes REST/MCP endpoints.
- **Dify** adds an optional **application shell** — web UI, batching, evaluation, and orchestration of multiple tools.

Both are **free to self-host**:
- LangFlow — MIT license.  
- Dify — Community Edition (Apache-2.0 + extra clauses) permitted for personal use.  

---

## 3. Repository Refactor Plan

### 3.1 Proposed directory layout

```
src/imageworks/
  core/
    image_io.py
    color_spaces.py
    geometry.py
    embeddings.py
    metrics.py
    types.py
  domains/
    mono/
      residual.py
      prompts.py
    narrate/
      regions.py
      prompts.py
    similarity/
      compare.py
      prompts.py
  adapters/
    langflow_nodes/
      preprocess.py
      metadata_reader.py
      mono_node.py
      narrate_node.py
      clip_node.py
      similarity_nodes.py
      formatters.py
  flows/
      mono.flow.json
      narrate.flow.json
      similarity.flow.json
  cli/
      main.py  (optional CLI wrappers)
tests/
```

### 3.2 Core dataclasses (`core/types.py`)
These standardise the interface between your algorithms and the orchestration layer.

```python
@dataclass
class ImageData:
    rgb: np.ndarray
    path: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None

@dataclass
class ResidualResult:
    residual_fraction: float
    dominant_colour: str
    hotspots: List[Dict[str, int]]
    stats: Dict[str, float]

@dataclass
class NarrateTopColor:
    name: str
    area_pct: float
    location_phrase: str

@dataclass
class NarrateResult:
    top_colors: List[NarrateTopColor]

@dataclass
class SimilarityMetrics:
    clip: float
    lpips: float
    ssim: float
    color_drift: str
    overlap_pct: float

@dataclass
class SimilarityVerdict:
    bucket: str
    similarity: float
    metrics: SimilarityMetrics
```

---

## 4. Domain Logic (cleaned up)

Each domain module isolates the analytical logic.  
You can paste existing code here and improve internally later.

### 4.1 Mono — residual colour detection

```python
# src/imageworks/domains/mono/residual.py
from imageworks.core.types import ImageData, ResidualResult
import numpy as np

def detect_residual_colour(img: ImageData, epsilon: float = 0.015) -> ResidualResult:
    """Analyse a monochrome image for unwanted colour."""
    rgb = img.rgb
    # TODO: replace with your HSV/Lab residual maths
    residual_fraction = 0.02
    dominant_colour = "warm orange"
    hotspots = [{"x":600, "y":120, "w":96, "h":210}]
    stats = {"p95_sat":0.06, "lab_a_mean":3.1, "lab_b_mean":7.2, "epsilon":epsilon}
    return ResidualResult(residual_fraction, dominant_colour, hotspots, stats)
```

### 4.2 Narrate — colour + location summary

```python
# src/imageworks/domains/narrate/regions.py
from imageworks.core.types import ImageData, NarrateResult, NarrateTopColor

def extract_top_colours_with_locations(img: ImageData, k: int = 6) -> NarrateResult:
    """Segment/cluster image → top colours with approximate locations."""
    top = [
        NarrateTopColor("teal", 36.0, "lower third"),
        NarrateTopColor("amber", 8.0, "frame edges"),
    ]
    return NarrateResult(top)
```

### 4.3 Similarity — metrics and fusion

```python
# src/imageworks/domains/similarity/compare.py
from imageworks.core.types import ImageData, SimilarityMetrics, SimilarityVerdict

def compute_similarity_metrics(a: ImageData, b: ImageData) -> SimilarityMetrics:
    """Return combined perceptual metrics (CLIP, LPIPS, SSIM, etc.)."""
    return SimilarityMetrics(clip=0.94, lpips=0.11, ssim=0.83, color_drift="warm", overlap_pct=0.72)

def fuse_metrics_to_bucket(m: SimilarityMetrics) -> SimilarityVerdict:
    if m.clip > 0.92 and m.ssim > 0.8 and m.lpips < 0.15:
        bucket = "crop" if m.overlap_pct < 0.85 else "tonal"
    elif m.clip < 0.70:
        bucket = "different"
    else:
        bucket = "colour"
    score = max(0, min(1, 0.5*m.clip + 0.3*m.ssim + 0.2*(1-m.lpips)))
    return SimilarityVerdict(bucket=bucket, similarity=score, metrics=m)
```

---

## 5. LangFlow Integration

### 5.1 Why LangFlow
- Visual orchestration and inspection of multi-step pipelines.  
- One-click serving of flows as REST endpoints.  
- Supports custom Python nodes (your domain modules).  

### 5.2 Environment Setup

```bash
# Recommended: Python 3.11+
pipx install langflow
langflow run --backend-only --host 0.0.0.0 --port 7860
# UI available at http://localhost:7860
```

### 5.3 Custom node registration

Each node wraps one of your domain functions.

Example:  
`src/imageworks/adapters/langflow_nodes/mono_node.py`
```python
from imageworks.core.types import ImageData
from imageworks.domains.mono.residual import detect_residual_colour

class MonoResidualDetectorNode:
    def __init__(self, epsilon: float = 0.015):
        self.epsilon = float(epsilon)
    def run(self, img: ImageData):
        r = detect_residual_colour(img, epsilon=self.epsilon)
        return {
            "residual_fraction": r.residual_fraction,
            "dominant_colour": r.dominant_colour,
            "hotspots": r.hotspots,
            "stats": r.stats,
            "epsilon": self.epsilon
        }
```

Repeat similar wrappers for **NarrateRegionsNode**, **SimilarityMetricsNode**, etc.

Register each as a *Custom Component* inside LangFlow (Settings → Custom Components).

### 5.4 Preprocess & Metadata nodes
Generic loader/normaliser using OpenCV; metadata stub can later use ExifTool.

### 5.5 Flow designs

Each flow is provided as a `.flow.json` in `src/imageworks/flows/`.

| Flow | Purpose | Inputs | Outputs |
|------|----------|---------|----------|
| **Mono** | Detect residual colour; output 1-sentence comment + score | Image | JSON (`is_mono_like`, `risk_score_20`, `statement`, etc.) |
| **Narrate** | Describe colours and location | Image | `caption`, `colors[]` |
| **Similarity** | Compare two images | Image A, B | `bucket`, `similarity`, `explanation`, `metrics` |

Each includes an **LLM node** (e.g. GPT-4o-mini, local OpenAI-compatible endpoint) with concise prompt templates.

### 5.6 Serving flows as REST APIs

Each flow, once imported, can be served:

```bash
langflow run --backend-only
```

Endpoints (example):

```
POST http://localhost:7860/api/v1/run/<FLOW_ID>
{
  "input": {"image_url": "file:///path/to/img.jpg"}
}
```

Responses are structured JSON objects ready for ingestion by your `process_photos.py`.

---

## 6. Optional Dify Layer

### 6.1 Purpose
Dify adds an **application surface** over your LangFlow services.

Use it when you need:
- **A browser UI** for curators/clients to run flows.  
- **Batch jobs / dataset evaluation** with retries & logging.  
- **Authentication, sharing, and analytics**.  
- **Workflow composition** (chain flows or add retrieval/code nodes).

If you’re the only user, you can skip Dify; LangFlow alone suffices.

### 6.2 Setup (Community Edition)

```bash
git clone https://github.com/langgenius/dify.git
cd dify/docker
cp .env.example .env
docker compose up -d
# visit http://localhost/install
```

Runs: Postgres + Redis + Dify API + Web UI + Worker.

### 6.3 Register LangFlow tools

In Dify UI → **Tools → Add HTTP Tool**

| Tool | URL | Method | Example body |
|------|-----|---------|--------------|
| `mono_flow` | `http://langflow:7860/api/v1/run/<MONO_FLOW_ID>` | POST | `{"input":{"image_url":"{{image_url}}"}}` |
| `narrate_flow` | `.../<NARRATE_FLOW_ID>` | POST | `{"input":{"image_url":"{{image_url}}"}}` |
| `similarity_flow` | `.../<SIMILARITY_FLOW_ID>` | POST | `{"input":{"image_a_url":"{{image_a_url}}","image_b_url":"{{image_b_url}}"}}` |

Add `x-api-key` header if LangFlow auth is enabled.

### 6.4 Workflow structure

```
[Start]
   |
   v
[Classifier: mode = mono/narrate/similarity]
   |--- mono ---> Tool: mono_flow
   |--- narrate -> Tool: narrate_flow
   |--- similarity -> Tool: similarity_flow
        |
        v
 [LLM: tidy response or summarise JSON]
        |
        v
      [End]
```

Optional: wrap in **Iteration node** to run a batch of images for competitions.

### 6.5 When to use Dify

| Scenario | Recommendation |
|-----------|----------------|
| You need a UI for non-developers | ✅ Use Dify |
| You want evaluation datasets and prompt versioning | ✅ Use Dify |
| Fully offline / local scripts only | ❌ LangFlow only |

---

## 7. Example Docker Compose (LangFlow + Dify)

```yaml
version: "3.8"
services:
  langflow:
    image: ghcr.io/langflow-ai/langflow:latest
    command: ["langflow","run","--backend-only","--host","0.0.0.0","--port","7860"]
    environment:
      - LANGFLOW_ALLOW_ORIGINS=*
    ports:
      - "7860:7860"
    restart: unless-stopped

  dify-stack:
    # Use Dify’s provided compose in dify/docker for full stack (api, web, db, redis)
    depends_on: [langflow]
    restart: unless-stopped
```

---

## 8. API Contract Examples

### Mono
Request:
```json
POST /api/v1/run/<FLOW_ID>
{ "input": { "image_url": "file:///shot.jpg" } }
```
Response:
```json
{
  "is_mono_like": false,
  "risk_score_20": 14,
  "statement": "A warm orange cast is visible along the right window mullions.",
  "dominant_colour": "warm orange",
  "residual_fraction": 0.021,
  "hotspots": [{"x":602,"y":118,"w":96,"h":210}]
}
```

### Narrate
```json
{
  "caption": "Muted teal washes across the lower third, with amber accents along the frame edges.",
  "colors": [{"name":"teal","area_pct":36,"location":"lower third"}]
}
```

### Similarity
```json
{
  "similarity": 0.92,
  "bucket": "crop",
  "explanation": "Same composition; tighter crop on the left with higher contrast.",
  "metrics": {"clip":0.94,"lpips":0.11,"ssim":0.83,"color_drift":"warm","overlap_pct":0.72}
}
```

---

## 9. Implementation Roadmap

| Stage | Task | Outcome |
|-------|------|----------|
| **1** | Refactor repo per §3 | Clean modular codebase |
| **2** | Implement stubs in `domains/` with existing logic | Stable functional layer |
| **3** | Add custom LangFlow nodes | Code accessible as components |
| **4** | Import & test flows (Mono/Narrate/Similarity) | Working REST endpoints |
| **5** | Optionally deploy Dify CE & connect tools | Browser UI & batching |
| **6** | (Later) Add dataset evaluation & comparison | Prompt/threshold tuning |

---

## 10. Key Benefits of the New Architecture

| Area | Improvement |
|------|--------------|
| **Code maintainability** | Domain-specific modules, dataclasses, clear I/O contracts |
| **Reusability** | Each analysis callable via LangFlow API or CLI |
| **Transparency** | Node-level inspection and tracing in LangFlow |
| **Scalability** | Batch & parallel execution through Dify Iteration |
| **User experience** | Optional Dify web app for non-technical collaborators |
| **Deployment** | Entire stack self-hosted, no paid dependencies |

---

## 11. Next Steps for the Assignee

1. Clone current `imageworks` repo to a feature branch (`feature/langflow-refactor`).  
2. Apply directory restructure (§3).  
3. Paste or port existing logic into new domain modules.  
4. Implement custom node wrappers under `adapters/langflow_nodes/`.  
5. Install LangFlow locally, import provided `flows/*.flow.json`, verify runs.  
6. Expose via REST and test from `process_photos.py`.  
7. (Optional) Deploy Dify CE via Docker and register the three tools.  
8. Document environment in `docs/dev-env/langflow-dify-setup.md`.

---

## 12. References

- **LangFlow Docs:** https://docs.langflow.org  
- **LangFlow GitHub:** https://github.com/langflow-ai/langflow  
- **Dify Docs:** https://docs.dify.ai  
- **Dify GitHub:** https://github.com/langgenius/dify  

---

### ✅ Deliverables Checklist
- [ ] Updated `imageworks/` package layout  
- [ ] Domain modules cleaned (mono, narrate, similarity)  
- [ ] Custom LangFlow node classes implemented  
- [ ] Three flows imported & tested in LangFlow  
- [ ] `flows/` folder committed  
- [ ] (Optional) Dify stack running with connected tools  
- [ ] Documentation file (`docs/langflow_dify_implementation.md`) generated for internal wiki  

---

*End of Document*
