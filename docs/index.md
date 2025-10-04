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
- [Model Registry Notes](domains/personal-tagger/model-registry.md)

## How-To Guides
- [AI Models and Prompting](guides/ai-models-and-prompting.md)
- [IDE Setup (WSL/VSCode)](guides/ide-setup-wsl-vscode.md)

## Runbooks & Operational Playbooks
- [Model Naming & Ollama Lingering Analysis](runbooks/model-naming-and-ollama-lingering.md)
- [Ollama Summary and Actions](runbooks/ollama-summary-and-actions.md)
- [OpenWebUI Setup](runbooks/openwebui-setup.md)
- [vLLM Deployment Guide](runbooks/vllm-deployment-guide.md)

## Reference Material
- [Model Downloader Guide](reference/model-downloader.md)

## Specifications & Proposals
- [Imageworks Specification](spec/imageworks-specification.md)
- [Color Narrator Specification](spec/imageworks-colour-narrator-specification.md)
- [Imageworks Colour Narrator Design](spec/design/imageworks-colour-narrator-design.md)
- [Proposals & Future Work](proposals/)

## Decisions & Gaps
- Formal architecture decision records (ADRs) have not been captured yet. The
  runbooks above contain context for naming and registry choices; converting
  these into ADRs is recommended.
- Personal Tagger lacks end-to-end workflow and operations documentation. Once
  the feature stabilises, add a domain overview and runbook alongside the existing
  registry notes.

## Using This Map
Start with the domain guides for workflow overviews, follow the architecture
section for subsystem design, apply runbooks during operations, and consult the
reference section for API or CLI specifics. Proposals house in-flight ideas and
research that may evolve into future specs.
