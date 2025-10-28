# Documentation Restructure Summary

This summary captures the May 2025 reorganisation of the ImageWorks documentation
set.

## New taxonomy
- Created the `environment/`, `reference/`, `runbooks/`, and `unsure/` top-level
  directories to segment workstation setup, technical references, operational
  guides, and pending triage items.【F:docs/index.md†L1-L31】
- Updated `docs/index.md` with quick links and a module coverage matrix so every
  component clearly lists its reference and runbook pairing.【F:docs/index.md†L13-L31】

## Module coverage
- Added fresh reference docs for color narrator, mono checker, image similarity
  checker, personal tagger, GUI control center, and ZIP extractor, aligning
  descriptions with the current source tree.【F:docs/reference/color-narrator.md†L1-L55】【F:docs/reference/mono-checker.md†L1-L36】【F:docs/reference/image-similarity-checker.md†L1-L40】【F:docs/reference/personal-tagger.md†L1-L40】【F:docs/reference/gui.md†L1-L40】【F:docs/reference/zip-extract.md†L1-L36】
- Authored matching runbooks for each module that outline launch commands,
  review steps, and troubleshooting tips.【F:docs/runbooks/color-narrator.md†L1-L70】【F:docs/runbooks/mono-checker.md†L1-L70】【F:docs/runbooks/image-similarity-checker.md†L1-L70】【F:docs/runbooks/personal-tagger.md†L1-L70】【F:docs/runbooks/gui.md†L1-L70】【F:docs/runbooks/zip-extract.md†L1-L70】【F:docs/runbooks/model-loader.md†L1-L80】【F:docs/runbooks/model-downloader.md†L1-L80】【F:docs/runbooks/chat-proxy.md†L1-L90】

## Legacy material consolidation
- Moved architecture diagrams, ADRs, implementation reports, and domain primers
  under `reference/` subfolders to keep historical knowledge accessible without
  mixing it with operational guides.【F:docs/reference/architecture/layered-registry.md†L1-L160】【F:docs/reference/history/phase-1-completion-report.md†L1-L40】
- Archived previous GUI user guides and background-service notes into
  `reference/history/` alongside phase completion reports for future research.【F:docs/reference/history/GUI-USER-GUIDE.md†L1-L40】

## Environment refresh
- Gathered workstation setup, model prompting advice, and LLM generation
  parameter guides into `environment/` for quick onboarding.【F:docs/environment/ide-setup-wsl-vscode.md†L1-L80】【F:docs/environment/ai-models-and-prompting.md†L1-L80】【F:docs/environment/llm-generation-parameters.md†L1-L60】

Use the new structure when adding documentation: update the relevant module
reference and runbook together, log proposals in `proposals/`, and stage uncertain
content in `unsure/` until it is triaged.
