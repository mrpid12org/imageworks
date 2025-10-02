import pathlib

DOC = pathlib.Path("docs/model-loader-overview.md")

REQUIRED_HEADINGS = [
    "# Model Loader Architecture & Integration Guide",
    "## 2. Core Data Structures",
    "## 4. Download Integration (Adapter Layer)",
    "## 6. Hashing & Integrity",
    "### 11.2 Lifecycle (Mermaid)",
]


def test_model_loader_overview_contains_required_sections():
    if not DOC.exists():
        raise AssertionError("model-loader-overview.md missing")
    text = DOC.read_text(encoding="utf-8")
    for h in REQUIRED_HEADINGS:
        assert h in text, f"Missing heading: {h}"
