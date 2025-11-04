# ImageWorks Documentation Index

The documentation set is now organised into four high-level categories:

- **Environment** – Workstation setup, tooling prerequisites, and global model
  guidance.
- **Reference** – Technical deep dives for every module plus historical
  architecture, specs, and decisions.
- **Runbooks** – Step-by-step operational guides for CLI and GUI workflows.
- **Proposals** – Forward-looking plans and feature ideas awaiting execution.
- **Unsure** – Temporary holding area when a document still needs reclassification.

## Quick links
- Environment: [`docs/environment/`](environment/)
- Reference (module overviews, architecture, specs): [`docs/reference/`](reference/)
- Runbooks: [`docs/runbooks/`](runbooks/)
- Proposals: [`docs/proposals/`](proposals/)
- Unsure: [`docs/unsure/`](unsure/)

## Module coverage
Each ImageWorks module now has at least one reference document and one runbook:

| Module | Reference | Runbook |
| --- | --- | --- |
| Chat proxy | [`reference/chat-proxy.md`](reference/chat-proxy.md) | [`runbooks/chat-proxy.md`](runbooks/chat-proxy.md) |
| Color narrator | [`reference/color-narrator.md`](reference/color-narrator.md) | [`runbooks/color-narrator.md`](runbooks/color-narrator.md) |
| GUI control center | [`reference/gui.md`](reference/gui.md) | [`runbooks/gui.md`](runbooks/gui.md) |
| Image similarity checker | [`reference/image-similarity-checker.md`](reference/image-similarity-checker.md) | [`runbooks/image-similarity-checker.md`](runbooks/image-similarity-checker.md) |
| Model downloader | [`reference/model-downloader.md`](reference/model-downloader.md) | [`runbooks/model-downloader.md`](runbooks/model-downloader.md) |
| Model loader | [`reference/model-loader.md`](reference/model-loader.md) | [`runbooks/model-loader.md`](runbooks/model-loader.md) |
| Mono checker | [`reference/mono-checker.md`](reference/mono-checker.md) | [`runbooks/mono-checker.md`](runbooks/mono-checker.md) |
| Personal tagger | [`reference/personal-tagger.md`](reference/personal-tagger.md) | [`runbooks/personal-tagger.md`](runbooks/personal-tagger.md) |
| Judge Vision | [`proposals/judge-vision-spec.md`](proposals/judge-vision-spec.md) | [`runbooks/judge-vision.md`](runbooks/judge-vision.md) |
| ZIP extractor | [`reference/zip-extract.md`](reference/zip-extract.md) | [`runbooks/zip-extract.md`](runbooks/zip-extract.md) |

Additional background material (architecture diagrams, ADRs, specs, and
implementation reports) lives under `reference/` with subfolders for easy lookup.

## How to contribute
1. Decide whether your change is environment, reference, runbook, or proposal
   material. Use `unsure/` only if the document still needs triage.
2. Update the relevant module reference and runbook when behaviour or CLI options
   change.
3. Link back to this index when adding new documents so future readers can find
   them quickly.
