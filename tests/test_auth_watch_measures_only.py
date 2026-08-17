"""Die Messung zu #1184 darf NICHTS abweisen.

Befund 2026-08-17: `/ai/chatgpt` antwortet ohne jeden
Authorization-Header mit 200. Ein Schluesselzwang waere die
naheliegende Reaktion und wuerde jeden bestehenden Aufrufer auf einen
Schlag brechen — guide-api, automation-api, knowledge-api, swfme, die
MCP-Werkzeuge, Tschepps neuen Worker.

Deshalb zuerst nur MESSEN: Wer ruft heute ohne Schluessel? Erst mit
dieser Zahl ist die Entscheidung ueber den Zwang treffbar, und sie
gehoert Alexander.

Dieser Test haelt fest, dass die Messung eine Messung BLEIBT. Ein
spaeterer Umbau, der hier heimlich eine Abweisung einbaut, faellt auf.
"""

import inspect

import main


def test_middleware_weist_nichts_ab():
    quelle = inspect.getsource(main._log_auth_presence)
    # Kein Abweisungspfad: keine Antwort wird selbst erzeugt.
    for verboten in ("HTTPException", "JSONResponse", "status_code=401",
                     "status_code=403", "raise "):
        assert verboten not in quelle, (
            f"Die Messung enthaelt '{verboten}' — sie weist ab, statt zu "
            "messen. Der Zwang ist Alexanders Entscheidung, nicht die "
            "Nebenwirkung einer Zaehlung."
        )
    # Und sie reicht IMMER durch.
    assert "return await call_next(request)" in quelle


def test_messung_stoert_den_dienst_nicht():
    """Ein Fehler in der Zaehlung darf keine Anfrage kosten."""
    quelle = inspect.getsource(main._log_auth_presence)
    assert "except Exception:" in quelle
    # Der Durchreicher steht AUSSERHALB des try — sonst haette ein
    # Fehler in der Messung die Antwort verschluckt.
    kopf, _, rest = quelle.partition("except Exception:")
    assert "return await call_next(request)" not in kopf


def test_nur_der_anonyme_fall_wird_geschrieben():
    """Jede Anfrage zu protokollieren wuerde das Journal fluten und die
    interessante Zeile darin unsichtbar machen."""
    quelle = inspect.getsource(main._log_auth_presence)
    assert "if not (hat_bearer or hat_key):" in quelle
    assert "#1184" in quelle
