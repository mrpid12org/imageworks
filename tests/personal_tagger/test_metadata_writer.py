from pathlib import Path

from imageworks.apps.personal_tagger.core.metadata_writer import PersonalMetadataWriter
from imageworks.apps.personal_tagger.core.models import (
    GenerationModels,
    KeywordPrediction,
    PersonalTaggerRecord,
)


class RecordingWriter(PersonalMetadataWriter):
    """Test double that records ExifTool invocations without executing them."""

    def __init__(self) -> None:
        super().__init__(backup_originals=False, overwrite_existing=True)
        self.calls = []

    def _run_exiftool(self, command):  # type: ignore[override]
        self.calls.append(command)
        return None

    def _has_existing_metadata(self, image_path: Path) -> bool:  # type: ignore[override]
        return False


def test_writer_builds_expected_command(tmp_path):
    raw_path = tmp_path / "example.cr2"
    raw_path.write_bytes(b"RAW DATA")

    record = PersonalTaggerRecord(
        image=raw_path,
        keywords=[KeywordPrediction(keyword="forest trail", score=0.91)],
        caption="A dense forest path.",
        description="A narrow trail winds through dense evergreen trees with light filtering from above.",
        critique="Judges praise the mood but suggest leading lines to tighten the story.",
        backend="test",
        models=GenerationModels(
            caption="test/caption",
            keywords="test/keywords",
            description="test/description",
            critique="test/description",
        ),
    )

    writer = RecordingWriter()

    assert writer.write(raw_path, record) is True
    assert writer.calls, "Expected ExifTool command to be recorded"

    command = writer.calls[0]
    assert command[0] == "exiftool"
    assert "-overwrite_original" in command
    assert any("forest trail" in arg for arg in command)
    assert any("A dense forest path" in arg for arg in command)
