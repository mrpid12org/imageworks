"""Tests for URL analyzer shorthand handling."""

from imageworks.tools.model_downloader.url_analyzer import URLAnalyzer


class TestURLAnalyzerShorthand:
    """Validate owner/repo parsing without full URLs."""

    def test_analyze_owner_repo_shorthand(
        self,
        mock_requests_get,
        sample_hf_response,
        sample_file_list,
        sample_config_json,
    ):
        analyzer = URLAnalyzer()

        mock_requests_get(
            "https://huggingface.co/api/models/microsoft/DialoGPT-medium",
            sample_hf_response,
        )
        mock_requests_get(
            "https://huggingface.co/api/models/microsoft/DialoGPT-medium/tree/main",
            sample_file_list,
        )
        mock_requests_get(
            "https://huggingface.co/microsoft/DialoGPT-medium/raw/main/config.json",
            sample_config_json,
        )

        analysis = analyzer.analyze_url("microsoft/DialoGPT-medium")

        assert analysis.repository.owner == "microsoft"
        assert analysis.repository.repo == "DialoGPT-medium"
        assert analysis.repository.branch == "main"
        assert analysis.files["config"]

    def test_analyze_owner_repo_with_branch(
        self,
        mock_requests_get,
        sample_hf_response,
        sample_file_list,
        sample_config_json,
    ):
        analyzer = URLAnalyzer()

        mock_requests_get(
            "https://huggingface.co/api/models/microsoft/DialoGPT-medium",
            sample_hf_response,
        )
        mock_requests_get(
            "https://huggingface.co/api/models/microsoft/DialoGPT-medium/tree/dev",
            sample_file_list,
        )
        mock_requests_get(
            "https://huggingface.co/microsoft/DialoGPT-medium/raw/dev/config.json",
            sample_config_json,
        )

        analysis = analyzer.analyze_url("microsoft/DialoGPT-medium@dev")

        assert analysis.repository.branch == "dev"
        assert analysis.repository.owner == "microsoft"
        assert analysis.repository.repo == "DialoGPT-medium"
