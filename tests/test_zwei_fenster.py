"""Zwei Budgetfenster nebeneinander (#4831, Vertrag 2026-08-26).

Der Vertrag verlangt „3 € je Kalendertag **und** 50 € je Monat, das
engere greift". Mein Wächter kannte bis heute nur EIN Fenster, per
Umgebungsvariable gewählt — und diese Variable gilt prozessweit für
alle Profile derselben Instanz. Sie umzustellen hätte Alex' Monats-
budget in ein Tagesbudget verwandelt: genau die Fenster-Verwechslung
aus #1314, nur andersherum.

Deshalb ein **zweites** Fenster neben dem ersten, und zwar nur für
Profile, deren Grant ein `monthly_budget_eur` trägt. `None` heißt
kein Monatsfenster — das ist der Zustand aller bestehenden Profile.
"""

import pytest

from ai.services import realtime_budget_guard as g


@pytest.fixture(autouse=True)
def frisch(tmp_path, monkeypatch):
    monkeypatch.setattr(g, "RESERVATIONS_PATH", tmp_path / "res.json")
    assert "/var/lib" not in str(g.RESERVATIONS_PATH)
    monkeypatch.delenv(g.SESSION_BUDGET_ENV, raising=False)


def _reserve(**kw):
    args = dict(profile_id="p", user_id="u", voice_session_id="vs1",
                max_parallel_sessions=5, daily_budget_eur=100.0)
    args.update(kw)
    return g.reserve_mint(**args)


# ───────────────────────────────── None heisst KEIN Monatsfenster

def test_ohne_monatslimit_aendert_sich_nichts():
    """Der Zustand aller bestehenden Profile. Ein zweites Fenster darf
    sie nicht rückwirkend beschneiden."""
    g.confirm_usage_charge(profile_id="p", user_id="u",
                           voice_session_id="vs0", cost_eur=99.0)
    assert _reserve(monthly_budget_eur=None) is not None


def test_null_ist_nicht_dasselbe_wie_keine_grenze():
    """`0.0` ist eine echte Null-Grenze, `None` ist gar keine.

    Wer beides gleichsetzt, öffnet entweder ein gesperrtes Profil oder
    sperrt ein offenes — dieselbe Dreiwertigkeit wie „markenoffen ist
    nicht unbekannt".
    """
    with pytest.raises(g.BudgetExceeded):
        _reserve(monthly_budget_eur=0.0)
    assert _reserve(monthly_budget_eur=None) is not None


# ───────────────────────────────────────── das engere Limit greift

def test_monatsgrenze_sperrt_obwohl_der_tag_frei_waere():
    g.confirm_usage_charge(profile_id="p", user_id="u",
                           voice_session_id="vs0", cost_eur=50.0)
    # Tag: 50 von 100 verbraucht, viel Luft. Monat: 50 von 50, voll.
    with pytest.raises(g.BudgetExceeded) as exc:
        _reserve(daily_budget_eur=100.0, monthly_budget_eur=50.0)
    assert exc.value.public_fields["window"] == "monthly"
    assert exc.value.public_fields["limit_eur"] == 50.0


def test_die_monatsabsage_zeigt_auf_den_monatsersten():
    """Der Kern von #1314, hier für das zweite Fenster.

    Stünde im Feld `window` das Hauptfenster, sagte die Absage
    „daily" — und der Client verspräche „morgen wieder", während der
    Topf erst am Monatsersten aufgeht.
    """
    g.confirm_usage_charge(profile_id="p", user_id="u",
                           voice_session_id="vs0", cost_eur=50.0)
    with pytest.raises(g.BudgetExceeded) as exc:
        _reserve(daily_budget_eur=100.0, monthly_budget_eur=50.0)
    f = exc.value.public_fields
    assert f["window"] == "monthly"
    assert f["resets_at"].split("T")[0].endswith("-01")


def test_tagesgrenze_greift_weiterhin_wenn_sie_die_engere_ist():
    g.confirm_usage_charge(profile_id="p", user_id="u",
                           voice_session_id="vs0", cost_eur=3.0)
    with pytest.raises(g.BudgetExceeded) as exc:
        _reserve(daily_budget_eur=3.0, monthly_budget_eur=50.0)
    assert exc.value.public_fields["limit_eur"] == 3.0


def test_beide_toepfe_werden_gefuellt():
    g.confirm_usage_charge(profile_id="p", user_id="u",
                           voice_session_id="vs0", cost_eur=2.0)
    with g._locked_state() as state:
        pv = state["profiles"]["p"]
    assert pv["daily_total_eur"] == pytest.approx(2.0)
    assert pv["monthly_total_eur"] == pytest.approx(2.0)


# ──────────────────────────── Sitzungsdeckel je Profil

def test_profildeckel_schlaegt_den_globalen(monkeypatch):
    """Alex hat global 2,00 € gesetzt; der Demo-Zugang soll 0,50 €.

    Beides gleichzeitig ging vorher nicht — der Wert war prozessweit.
    """
    monkeypatch.setenv(g.SESSION_BUDGET_ENV, "2.00")
    monkeypatch.setenv(f"{g.SESSION_BUDGET_ENV}__PRODUCT_FINDER", "0.50")
    assert g._session_budget_eur("product-finder") == 0.5
    assert g._session_budget_eur("alex-default") == 2.0
    assert g._session_budget_eur(None) == 2.0


def test_bindestrich_im_profilnamen_wird_uebersetzt(monkeypatch):
    """Umgebungsvariablen tragen keine Bindestriche. Ohne diese
    Übersetzung wäre der Deckel für `product-finder` unsetzbar — und
    zwar lautlos: Er fiele auf den globalen Wert zurück."""
    monkeypatch.setenv(f"{g.SESSION_BUDGET_ENV}__PRODUCT_FINDER", "0.50")
    assert g._session_budget_eur("product-finder") == 0.5


def test_unsinn_im_profildeckel_faellt_auf_global_zurueck(monkeypatch):
    monkeypatch.setenv(g.SESSION_BUDGET_ENV, "2.00")
    monkeypatch.setenv(f"{g.SESSION_BUDGET_ENV}__PRODUCT_FINDER", "viel")
    assert g._session_budget_eur("product-finder") == 2.0


# ───────────────────────────────────── der Grant traegt das Feld

def test_grant_kennt_das_monatslimit_und_None_ist_die_vorgabe():
    import dataclasses

    from ai.services.realtime_grant_verifier import VerifiedGrant
    feld = {f.name: f for f in dataclasses.fields(VerifiedGrant)}
    assert "monthly_budget_eur" in feld
    assert feld["monthly_budget_eur"].default is None


def test_mint_reicht_das_monatslimit_durch():
    import inspect

    from ai.routes import realtime_routes as rr
    quelle = inspect.getsource(rr.mint_realtime_token)
    assert "monthly_budget_eur=grant.monthly_budget_eur" in quelle


# ─────────────────────────────────── Umbruch des Monatstopfs

def test_der_monatstopf_laeuft_am_monatsersten_leer(monkeypatch):
    """Ohne Umbruch traegt ein Profil den Verbrauch des Vormonats ewig
    mit — es waere ab dem Ersten dauerhaft gesperrt statt frei.

    Der Fehler wiegt schwerer als eine zu strenge Absage: er faellt
    nicht auf. Niemand meldet ein Budget, das zu frueh zu ist; man
    haelt den Dienst fuer kaputt.
    """
    monkeypatch.setattr(g, "_month_str", lambda: "2026-08")
    g.confirm_usage_charge(profile_id="p", user_id="u",
                           voice_session_id="vs0", cost_eur=50.0)
    with pytest.raises(g.BudgetExceeded):
        _reserve(monthly_budget_eur=50.0)

    monkeypatch.setattr(g, "_month_str", lambda: "2026-09")
    assert _reserve(monthly_budget_eur=50.0) is not None


def test_innerhalb_des_monats_bleibt_der_stand_stehen(monkeypatch):
    """Die Gegenrichtung. Ein Topf, der bei jedem Blick leerlaeuft,
    ist kein Budget — er sieht nur aus wie eines."""
    monkeypatch.setattr(g, "_month_str", lambda: "2026-08")
    for _ in range(3):
        g.confirm_usage_charge(profile_id="p", user_id="u",
                               voice_session_id="vs0", cost_eur=20.0)
    with g._locked_state() as state:
        assert state["profiles"]["p"]["monthly_total_eur"] == pytest.approx(60.0)
    with pytest.raises(g.BudgetExceeded) as exc:
        _reserve(monthly_budget_eur=50.0)
    assert exc.value.public_fields["window"] == "monthly"


def test_tagesumbruch_laesst_den_monatstopf_stehen(monkeypatch):
    """Die beiden Fenster sind unabhaengig. Loeschte der Tageswechsel
    den Monatsstand mit, waere die Monatsgrenze nie erreichbar."""
    monkeypatch.setattr(g, "_month_str", lambda: "2026-08")
    monkeypatch.setattr(g, "_today_str", lambda: "2026-08-25")
    g.confirm_usage_charge(profile_id="p", user_id="u",
                           voice_session_id="vs0", cost_eur=40.0)
    monkeypatch.setattr(g, "_today_str", lambda: "2026-08-26")
    with g._locked_state() as state:
        pv = g._profile_view(state, "p", g._today_str())
        assert pv["daily_total_eur"] == pytest.approx(0.0)
        assert pv["monthly_total_eur"] == pytest.approx(40.0)
