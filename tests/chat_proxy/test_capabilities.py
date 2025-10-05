from imageworks.chat_proxy.capabilities import supports_vision


class _Probe:
    def __init__(self, ok: bool) -> None:
        self.vision_ok = ok


class _Probes:
    def __init__(self, probe):
        self.vision = probe


def test_supports_vision_prefers_probe_over_capabilities():
    class Entry:
        probes = _Probes(_Probe(False))
        capabilities = {"vision": True}

    assert supports_vision(Entry()) is False


def test_supports_vision_falls_back_to_capabilities_when_unprobed():
    class Entry:
        probes = _Probes(None)
        capabilities = {"vision": True}

    assert supports_vision(Entry()) is True


def test_supports_vision_respects_absence():
    class Entry:
        probes = _Probes(None)
        capabilities = {"vision": False}

    assert supports_vision(Entry()) is False
