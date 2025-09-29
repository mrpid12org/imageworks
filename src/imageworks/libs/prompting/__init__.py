"""Shared prompt library utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Generic, Iterable, List, TypeVar, Union

__all__ = [
    "PromptProfileBase",
    "PromptLibrary",
]


@dataclass(frozen=True)
class PromptProfileBase:
    """Base metadata shared by prompt profiles."""

    id: int
    name: str
    description: str


TProfile = TypeVar("TProfile", bound=PromptProfileBase)


class PromptLibrary(Generic[TProfile]):
    """Registry of prompt profiles with lookup helpers."""

    def __init__(self, profiles: Dict[int, TProfile], *, default_id: int) -> None:
        if default_id not in profiles:
            raise ValueError(f"Default id {default_id} not found in profiles")
        self._profiles_by_id: Dict[int, TProfile] = dict(profiles)
        self._profiles_by_name: Dict[str, TProfile] = {
            profile.name: profile for profile in profiles.values()
        }
        if len(self._profiles_by_name) != len(self._profiles_by_id):
            raise ValueError("Profile names must be unique")
        self._default_id = default_id

    @property
    def default(self) -> TProfile:
        """Return the default profile."""

        return self._profiles_by_id[self._default_id]

    def get(self, identifier: Union[int, str, None]) -> TProfile:
        """Retrieve a profile by numeric id, name, or fallback to default."""

        if isinstance(identifier, int):
            return self._profiles_by_id.get(identifier, self.default)
        if isinstance(identifier, str):
            return self._profiles_by_name.get(identifier, self.default)
        return self.default

    def list(self) -> List[TProfile]:
        """Return all profiles sorted by id."""

        return sorted(self._profiles_by_id.values(), key=lambda profile: profile.id)

    def ids(self) -> Iterable[int]:
        """Return available profile ids."""

        return self._profiles_by_id.keys()

    def names(self) -> Iterable[str]:
        """Return available profile names."""

        return self._profiles_by_name.keys()
