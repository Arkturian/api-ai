"""Der Dialog-Auftrag darf die Ereignisschleife nicht blockieren.

Die Produktion laeuft als asyncio-Aufgabe im selben Prozess, der auch
`/ai/dialog/status` beantwortet. Jeder ffmpeg- oder ffprobe-Aufruf, der
DIREKT in einer Koroutine steht, haelt die ganze Schleife an — der
Auftrag laeuft gesund weiter, aber kein Statusabruf kommt durch.

Anlass (Story, 12.09.): dialog.php pollt den Status mit 8 s
Client-Timeout und gibt nach drei Fehlschlaegen auf. Die Frage war, ob
der Statusabruf laenger als 8 s brauchen kann. Konnte er: vier
subprocess.run-Aufrufe standen unverpackt in `_produce_audio_drama`,
darunter ein ffmpeg-concat ueber die volle Musikdatei und eine Sonde je
Sprechzeile.

Dieser Test haelt die Regel strukturell fest, weil die Verletzung im
Betrieb nur als sporadischer Client-Abbruch sichtbar waere — also
genau dann, wenn niemand hinschaut.
"""

import ast
import io

import pytest

_DATEIEN = (
    "ai/services/audio_drama_service.py",
    "ai/routes/dialog_routes.py",
)

_VERBOTEN = {("subprocess", "run"), ("subprocess", "check_output"),
             ("subprocess", "call"), ("time", "sleep")}


def _blockierende_aufrufe(pfad):
    """Blockierende Aufrufe, die unmittelbar in einer Koroutine stehen.

    Verschachtelte gewoehnliche Funktionen zaehlen NICHT: sie sind der
    zulaessige Weg, wenn sie ueber `asyncio.to_thread` gerufen werden.
    """
    baum = ast.parse(io.open(pfad, encoding="utf-8").read())
    treffer = []

    def kern(knoten, in_koroutine):
        for kind in ast.iter_child_nodes(knoten):
            if isinstance(kind, ast.AsyncFunctionDef):
                kern(kind, True)
                continue
            if isinstance(kind, (ast.FunctionDef, ast.Lambda)):
                kern(kind, False)
                continue
            if in_koroutine and isinstance(kind, ast.Call):
                f = kind.func
                if (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name)
                        and (f.value.id, f.attr) in _VERBOTEN):
                    treffer.append(f"{pfad}:{kind.lineno} {f.value.id}.{f.attr}")
            kern(kind, in_koroutine)

    kern(baum, False)
    return treffer


@pytest.mark.parametrize("pfad", _DATEIEN)
def test_keine_blockierenden_aufrufe_in_koroutinen(pfad):
    treffer = _blockierende_aufrufe(pfad)
    assert not treffer, (
        "Blockierender Aufruf direkt in einer Koroutine — ueber "
        "asyncio.to_thread fuehren:\n  " + "\n  ".join(treffer)
    )


def test_der_waechter_wuerde_den_echten_fehler_finden():
    """Gegenprobe: der Wachhund darf nicht bloss immer schweigen."""
    quelle = '''
import subprocess

async def produziere():
    subprocess.run(["ffmpeg"])

def sonde():
    subprocess.run(["ffprobe"])
'''
    import tempfile
    import os
    fd, pfad = tempfile.mkstemp(suffix=".py")
    os.write(fd, quelle.encode())
    os.close(fd)
    try:
        treffer = _blockierende_aufrufe(pfad)
        assert len(treffer) == 1, treffer
        assert "subprocess.run" in treffer[0]
    finally:
        os.unlink(pfad)
