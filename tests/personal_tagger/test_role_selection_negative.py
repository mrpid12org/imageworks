import pytest

from imageworks.model_loader.role_selection import select_by_role, CapabilityError


def test_select_by_role_missing_role():
    # choose a role unlikely to exist
    missing_role = "__nonexistent_role__"
    with pytest.raises(CapabilityError) as exc:
        select_by_role(missing_role)
    assert missing_role in str(exc.value)
