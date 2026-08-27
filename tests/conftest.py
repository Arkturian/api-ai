"""Gemeinsame Vorbedingungen für die Produktfinder-Tests.

Seit dem Kundenschnitt (2026-08-27) trägt der Kern **keine** Marken-
und Kategorienliste mehr; beide kommen zur Laufzeit aus der
Kunden-API. Tests, die Werkzeugschema oder Markenprüfung untersuchen,
brauchen deshalb eine Quelle — sonst prüfen sie den fail-closed-Pfad
statt des Verhaltens, das sie beschreiben.

Die Werte hier sind **Testdaten**, keine Kundendaten im Kern: Sie
liegen unter `tests/`, nicht unter `ai/`, und der Leak-Wächter schaut
ausdrücklich nur in `ai/`.
"""

import os

import pytest

os.environ.setdefault("OPENAI_API_KEY", "x")

TEST_MARKEN = ["O'Neal", "ONE Industries", "Kini Red Bull"]
TEST_KATEGORIEN = ["helmets-mx", "jerseys-offroad", "gloves", "goggles",
                   "boots-mx", "protection-mx"]


@pytest.fixture(autouse=True)
def _kundendaten(monkeypatch, request):
    """Liefert Marken und Kategorien, ohne die Kunden-API zu rufen.

    Tests, die ausdrücklich den Fall „nicht beschaffbar" prüfen wollen,
    heben das mit `@pytest.mark.ohne_kundendaten` auf.
    """
    from ai.routes import realtime_routes as rr
    rr._marken_cache["werte"] = None
    rr._kategorien_cache["werte"] = None
    if request.node.get_closest_marker("ohne_kundendaten"):
        yield
    else:
        monkeypatch.setattr(rr, "_oneal_marken", lambda: list(TEST_MARKEN))
        monkeypatch.setattr(rr, "_oneal_kategorien",
                            lambda: list(TEST_KATEGORIEN))
        yield
    rr._marken_cache["werte"] = None
    rr._kategorien_cache["werte"] = None


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "ohne_kundendaten: prueft den fail-closed-Pfad ohne Marken/Kategorien",
    )
