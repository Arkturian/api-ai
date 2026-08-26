"""Identitaeten in Logzeilen kuerzen, ohne sie zu verwechseln.

Anlass: AuthApi stellt seit `fe3e55b` Service-Principals aus, deren
`sub` die Form `agent:<name>` traegt. Die neun Logstellen kuerzten
mit `[:8]` — eine Regel, die fuer UUIDs stimmt und fuer Dienstnamen
genau das wegwirft, was sie unterscheidet.
"""

import re

import pytest

from ai.services.realtime_identity import kurz_id


def test_zwei_principals_bleiben_unterscheidbar():
    """Der eigentliche Fehler. Mit `[:8]` heissen beide `agent:pr` —
    ein Log, das zwei Verursacher gleich benennt, beantwortet die
    Frage falsch, statt sie offen zu lassen."""
    a = kurz_id("agent:productfinder-internal-demo")
    b = kurz_id("agent:productfinder-batch")
    assert a != b


def test_dienstkennung_bleibt_vollstaendig():
    assert kurz_id("agent:productfinder-internal-demo") == \
        "agent:productfinder-internal-demo"


def test_uuid_bleibt_gekuerzt():
    """Die Gegenrichtung: ein `sub`, das eine Person bezeichnet, darf
    NICHT vollstaendig ins Log. Wer die Kuerzung pauschal aufhebt,
    repariert die Lesbarkeit und bricht die Datensparsamkeit."""
    voll = "74e8e363-1234-5678-9abc-def012345678"
    assert kurz_id(voll) == "74e8e363"
    assert len(kurz_id(voll)) == 8


def test_leer_und_none_sagen_strich():
    """Nie `None` in eine Logzeile — und nie etwas, das mit einem
    echten Wert zu verwechseln waere."""
    assert kurz_id(None) == "-"
    assert kurz_id("") == "-"


def test_service_praefix_gilt_auch():
    assert kurz_id("service:oneal-bff") == "service:oneal-bff"


# ───────────────────────── die Aufrufstellen benutzen den Helfer

@pytest.mark.parametrize("pfad", [
    "ai/services/realtime_budget_guard.py",
    "ai/services/realtime_grant_verifier.py",
    "ai/routes/realtime_routes.py",
])
def test_keine_alte_kuerzung_mehr_im_code(pfad):
    """Ein zurueckgebliebenes `sub[:8]` faellt nicht auf — es
    produziert eine plausible Logzeile mit falschem Inhalt."""
    quelle = open(pfad).read()
    treffer = re.findall(r"\b(?:sub|user_id)\[:8\]", quelle)
    assert not treffer, f"{pfad}: {treffer}"


def test_der_waechter_deckt_den_fehlenden_import_ab():
    """`kurz_id` wird in Funktionsrumpfen benutzt. Fehlt der Import,
    laedt das Modul trotzdem und der NameError kommt erst im Betrieb —
    genau der Devlog-Fehler vom Vormittag."""
    import ai.routes.realtime_routes as rr
    import ai.services.realtime_budget_guard as bg
    import ai.services.realtime_grant_verifier as gv
    for modul in (rr, bg, gv):
        assert callable(getattr(modul, "kurz_id", None)), modul.__name__
