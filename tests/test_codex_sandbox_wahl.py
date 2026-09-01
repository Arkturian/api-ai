"""Codex-Sandbox-Wahl für /ai/chatgpt (Kleinhirn-Derivator, #4907 §14.5).

Der echte Fehler, gegen den diese Tests gehalten sind: Der Endpunkt lief
historisch IMMER mit `--dangerously-bypass-approvals-and-sandbox` — volles
Netz + Dateisystem, keine Rückfragen. Für Aufrufer, die FREMDEN Text durch
Codex leiten (der Derivator verdichtet beliebige Agent-Turns), ist jeder
Turn damit eine Injektionsfläche in eine werkzeugfähige CLI ohne Sandbox.

Drei Verträge:
1. Default bleibt das alte Verhalten — bestehende Aufrufer unberührt.
2. `sandbox="read-only"` ersetzt das Bypass-Flag durch `-s read-only` —
   und zwar ERSATZ, nicht Ergänzung: beide zugleich wäre wieder offen.
3. Unbekannte Werte fallen auf read-only, nie auf Vollzugriff — ein
   Tippfehler im einschränkenden Feld darf die Sandbox nicht öffnen.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai.routes.text_ai_routes import _codex_sandbox_args  # noqa: E402

BYPASS = "--dangerously-bypass-approvals-and-sandbox"


def test_default_bleibt_vollzugriff():
    assert _codex_sandbox_args(None) == [BYPASS]
    assert _codex_sandbox_args("") == [BYPASS]
    assert _codex_sandbox_args("   ") == [BYPASS]


def test_read_only_ersetzt_das_bypass_flag():
    args = _codex_sandbox_args("read-only")
    assert args == ["-s", "read-only"]
    assert BYPASS not in args


def test_unbekannter_wert_faellt_auf_read_only_nicht_auf_offen():
    for tippfehler in ("readonly", "strict", "READ-ONLY", "safe"):
        args = _codex_sandbox_args(tippfehler)
        assert BYPASS not in args, tippfehler
        assert args == ["-s", "read-only"], tippfehler
