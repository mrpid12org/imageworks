from imageworks.apps.judge_vision.prompts import get_prompt


def test_render_user_prompt_escapes_literal_braces():
    profile = get_prompt("club_judge_json")
    text = profile.render_user_prompt(
        title="Example",
        category="Open",
        notes="",
        caption="",
        keyword_preview="",
        compliance_findings="ok",
        technical_signals="ok",
    )
    assert "subscores" in text
    assert '"style"' in text
    assert "{{" not in text  # double braces replaced
    assert text.count("{") == text.count("}")
