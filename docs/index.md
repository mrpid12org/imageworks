# Imageworks Documentation Map

This index groups the documentation set into consistent categories so teams can
quickly locate design references, domain guides, and operational procedures.

## Architecture & System Design
- [Deterministic Model Serving](architecture/deterministic-model-serving.md)
- [Model Loader Architecture & Integration](architecture/model-loader-overview.md)
- [Layered Registry](architecture/layered-registry.md)
- [Personal Tagger Architecture Overview](architecture/personal-tagger-architecture-overview.md)
- [Project Structure](architecture/project-structure.md)

## Domain Guides
### Mono (Competition Checker)
- [Mono Overview](domains/mono/mono-overview.md)
- [Mono Workflow](domains/mono/mono-workflow.md)
- [Mono Technical Deep Dive](domains/mono/mono-technical.md)
- [Processing Downloads Pipeline](domains/mono/processing-downloads.md)

### Color Narrator
- [Color Narrator Reference](domains/color-narrator/reference.md)

### Personal Tagger
- [Overview](domains/personal-tagger/overview.md)
- [Model Registry Notes](domains/personal-tagger/model-registry.md)

## How-To Guides
- [AI Models and Prompting](guides/ai-models-and-prompting.md)
- [IDE Setup (WSL/VSCode)](guides/ide-setup-wsl-vscode.md)
- [Image Similarity Checker](guides/image-similarity-checker.md)

## Runbooks & Operational Playbooks
- [Model Naming & Ollama Lingering Analysis](runbooks/model-naming-and-ollama-lingering.md)
- [Ollama Summary and Actions](runbooks/ollama-summary-and-actions.md)
- [OpenWebUI Setup](runbooks/openwebui-setup.md)
- [vLLM Deployment Guide](runbooks/vllm-deployment-guide.md)

## Reference Material
- [Model Downloader Guide](reference/model-downloader.md)
- [Chat Proxy (OpenAI-compatible)](reference/chat-proxy.md)
- [Grounding Duplicate Detection Decisions](analysis/image-similarity-grounding.md)

## Specifications & Proposals
- [Imageworks Specification](spec/imageworks-specification.md)
- [Color Narrator Specification](spec/imageworks-colour-narrator-specification.md)
- [Imageworks Colour Narrator Design](spec/design/imageworks-colour-narrator-design.md)
- [Proposals & Future Work](proposals/)

## Decisions & Gaps
- [ADR 0001 – Unified Model Identity and Registry Hygiene](decisions/0001-model-naming-and-registry.md)
- Personal Tagger still needs dedicated operational runbooks (e.g. tips for XMP
  integration, Lightroom workflows) to sit alongside the new overview and
  registry notes.

## Using This Map
Start with the domain guides for workflow overviews, follow the architecture
section for subsystem design, apply runbooks during operations, and consult the
reference section for API or CLI specifics. Proposals house in-flight ideas and
research that may evolve into future specs.
