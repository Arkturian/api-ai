"""#3165: codex als eigener Dienstbenutzer apiai-codex mit eigenen Konten.

- Wrapper codex-als-dienst: Konto als erstes Argument, gleiche Optionsliste und
  Werkzeugsperren wie codex-als-alex, nichts aus /home/alex.
- Kontowahl wie cloud-api codex_konto_wechsel: fast voll ab 95 %, Ziel mit
  >= 20 Punkten Vorsprung, 1 h Mindestabstand; Ereignisse fuer Login.
- Zustand nur in der Datei (uvicorn --workers 2).
"""
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from ai.services import cli_dienst, cli_katalog, codex_konten as ck

REPO = Path(__file__).resolve().parents[1]
WRAPPER = "/usr/local/libexec/api-ai"
E = "/var/lib/api-ai-cli/eingang"


# --- Wrapper -----------------------------------------------------------------

def _wrapper(tmp_path):
    konten = tmp_path / "konten"
    for k in ("apopovic", "tsaier"):
        (konten / k).mkdir(parents=True)
    (konten / "apopovic" / "config.toml").write_text(
        'model = "gpt-6.1-terra"\nmodel_reasoning_effort = "high"\n[mcp_servers.x]\nmodel = "nein"\n')
    s = (REPO / "deploy" / "apiai" / "codex-als-dienst").read_text()
    s = s.replace("KONTEN=/var/lib/api-ai-codex/konten", f"KONTEN={konten}")
    s = s.replace("  exec env -i", "  echo START env -i")
    p = tmp_path / "codex-als-dienst"
    p.write_text(s)
    p.chmod(0o755)
    return p, konten


def _lauf(p, *args, stdin="hallo"):
    return subprocess.run([str(p), *args], input=stdin, capture_output=True, text=True)


ECHT = [
    ["apopovic", "text", "exec", "--json", "--skip-git-repo-check", "-c", "features.shell_tool=false",
     "--ephemeral", "--model", "gpt-6.1-terra", "-c", "model_reasoning_effort=high", "-i", f"{E}/a.jpg", "--", "-"],
    ["tsaier", "text", "exec", "--json", "-s", "read-only", "-C", "/var/lib/api-ai-cli/arbeit/codex-imagegen-x", "-"],
    ["apopovic", "haupt", "exec", "--json", "-"],
    ["tsaier", "text", "debug", "models"],
    ["apopovic", "text", "--version"],
]
FEINDLICH = [
    ["../apopovic", "text", "exec", "-"],
    ["Apopovic", "text", "exec", "-"],
    ["fehlt", "text", "exec", "-"],
    ["apopovic", "exec", "-"],
    ["apopovic", "text", "exec", "--dangerously-bypass-approvals-and-sandbox", "-"],
    ["apopovic", "text", "exec", "-c", "features.shell_tool=true", "-"],
    ["apopovic", "text", "exec", "-c", "sandbox_mode=danger-full-access", "-"],
    ["apopovic", "text", "exec", "-s", "workspace-write", "-"],
    ["apopovic", "text", "exec", "-i", "/home/alex/.ssh/id_ed25519", "--", "-"],
    ["apopovic", "text", "exec", "-C", "/", "-"],
    ["apopovic", "text", "exec", "--json", "mach was"],
    ["apopovic", "text", "mcp", "list"],
    ["apopovic", "text", "login"],
    ["apopovic", "text", "app-server"],
    ["apopovic", "limits", "x"],
    ["apopovic", "konfig", "x"],
]


@pytest.mark.parametrize("args", ECHT)
def test_wrapper_laesst_echte_aufrufe_durch(tmp_path, args):
    p, konten = _wrapper(tmp_path)
    r = _lauf(p, *args)
    assert r.returncode == 0, r.stderr
    assert r.stdout.startswith("START env -i HOME=/var/lib/api-ai-codex ")
    assert f"CODEX_HOME={konten}/{args[0]} " in r.stdout
    assert "/home/alex" not in r.stdout


@pytest.mark.parametrize("args", FEINDLICH)
def test_wrapper_weist_angriffe_ab(tmp_path, args):
    p, _ = _wrapper(tmp_path)
    r = _lauf(p, *args)
    assert r.returncode == 64 and "START" not in r.stdout, (args, r.stdout)


def test_wrapper_erzwingt_read_only_und_werkzeuge_aus(tmp_path):
    p, _ = _wrapper(tmp_path)
    r = _lauf(p, "apopovic", "haupt", "exec", "--json", "-")
    kopf = r.stdout.split(" exec ", 1)[1]
    assert "-s read-only --json -" in kopf
    for f in ("shell_tool", "unified_exec", "plugins", "apps", "browser_use", "view_image", "computer_use"):
        assert f"-c features.{f}=false" in kopf
    assert "features.code_mode_host=false" not in kopf


def test_wrapper_sperrt_dasselbe_wie_der_alte_und_der_code():
    import re
    from ai.routes.text_ai_routes import CODEX_GESPERRT
    neu = re.search(r"GESPERRT=\(([^)]*)\)", (REPO / "deploy/apiai/codex-als-dienst").read_text()).group(1).split()
    alt = re.search(r"GESPERRT=\(([^)]*)\)", (REPO / "deploy/apiai/codex-als-alex").read_text()).group(1).split()
    assert set(neu) == set(alt) == set(CODEX_GESPERRT)


def test_wrapper_konfig_nur_modell_und_effort(tmp_path):
    p, _ = _wrapper(tmp_path)
    r = _lauf(p, "apopovic", "konfig")
    assert r.returncode == 0 and r.stdout.splitlines() == ['model = "gpt-6.1-terra"', 'model_reasoning_effort = "high"']
    assert _lauf(p, "tsaier", "konfig").stdout == ""


def test_wrapper_limits_schickt_nur_feste_anfragen():
    s = (REPO / "deploy/apiai/codex-als-dienst").read_text()
    teil = s.split("  limits)", 1)[1].split(";;", 1)[0]
    assert "exec 0</dev/null" in teil                 # Aufrufer-stdin geht nicht an den app-server
    assert "account/rateLimits/read" in teil and "account/read" in teil
    assert '"$@"' not in teil


def test_sudoers_kennt_den_neuen_wrapper():
    zeilen = [z for z in (REPO / "deploy/apiai/sudoers-api-ai-konten").read_text().splitlines()
              if z.startswith("apiai ALL=")]
    assert any("(apiai-codex)" in z and z.endswith(f"{WRAPPER}/codex-als-dienst") for z in zeilen)
    assert not any("(alex)" in z for z in zeilen)
    assert all("NOSETENV:" in z for z in zeilen)


# --- api-ai im Kontomodus ----------------------------------------------------

@pytest.fixture
def konto(monkeypatch, tmp_path):
    wurzel = tmp_path / "konten"
    monkeypatch.setenv("API_AI_CLI_WRAPPER_DIR", WRAPPER)
    monkeypatch.setenv("CODEX_RUN_AS_USER", "apiai-codex")
    monkeypatch.setenv("API_AI_CODEX_KONTEN", "apopovic,tsaier")
    monkeypatch.setenv("API_AI_CODEX_KONTEN_WURZEL", str(wurzel))
    monkeypatch.setenv("API_AI_CODEX_KONTEN_ZUSTAND", str(tmp_path / "zustand.json"))
    ck._zuruecksetzen()
    return SimpleNamespace(wurzel=wurzel, zustand=tmp_path / "zustand.json")


def test_ohne_konten_bleibt_der_alte_wrapper(monkeypatch):
    monkeypatch.setenv("API_AI_CLI_WRAPPER_DIR", WRAPPER)
    monkeypatch.delenv("API_AI_CODEX_KONTEN", raising=False)
    assert not cli_dienst.konto_modus()
    cmd = cli_dienst.codex_befehl("alex", ["codex", "exec", "-"], {"CODEX_HOME": "/home/alex/.codex"})
    assert cmd[4].endswith("/codex-als-alex")


def test_befehl_nimmt_das_aktive_konto(konto):
    cmd = cli_dienst.codex_befehl("apiai-codex", ["codex", "exec", "--json", "-"], {"CODEX_HOME": "/home/alex/.codex"})
    assert cmd == ["sudo", "-n", "-u", "apiai-codex", f"{WRAPPER}/codex-als-dienst", "apopovic", "text",
                   "exec", "--json", "-"]


def test_befehl_bleibt_beim_konto_aus_dem_home(konto):
    """imagegen sucht das Bild im Home, das es vorher gewaehlt hat: wechselt
    das Konto dazwischen, muss der Lauf trotzdem dort landen."""
    h = ck.home("tsaier")
    cmd = cli_dienst.codex_befehl("apiai-codex", ["codex", "exec", "-"], {"CODEX_HOME": h})
    assert cmd[5] == "tsaier"
    assert cli_dienst.codex_text_home() == ck.home("apopovic")


def test_konfig_ueber_wrapper_mit_konto(konto, monkeypatch):
    rufe = []
    monkeypatch.setattr(cli_dienst.subprocess, "run", lambda c, **k: rufe.append(c) or SimpleNamespace(
        returncode=0, stdout='model = "m1"\nmodel_reasoning_effort = "low"\n'))
    assert cli_katalog.codex_config(ck.home("tsaier")) == ("m1", "low")
    assert rufe == [["sudo", "-n", "-u", "apiai-codex", f"{WRAPPER}/codex-als-dienst", "tsaier", "konfig"]]


def test_aktives_konto_kommt_aus_der_datei(konto):
    """Zwei uvicorn-Prozesse: was der eine schreibt, sieht der andere sofort."""
    assert ck.aktiv() == "apopovic"
    konto.zustand.write_text(json.dumps({"aktiv": "tsaier"}))
    assert ck.aktiv() == "tsaier"
    konto.zustand.write_text(json.dumps({"aktiv": "unbekannt"}))
    assert ck.aktiv() == "apopovic"


# --- Messen und entscheiden --------------------------------------------------

def _antwort(pct, reset=1000, email="a@b", fehler=None, erreicht=False):
    acc = {"id": 2, "result": {"account": {"email": email}}}
    rl = {"id": 3, "result": {"rateLimits": {"primary": {"usedPercent": pct, "resetsAt": reset},
                                             "rateLimitReachedType": "x" if erreicht else None}}}
    if fehler:
        acc = {"id": 2, "error": {"message": fehler}}
        rl = {"id": 3, "error": {"message": fehler}}
    return json.dumps(acc) + "\n" + json.dumps(rl) + "\n"


def test_auswerten():
    assert ck.auswerten(_antwort(40))["state"] == "ok"
    assert ck.auswerten(_antwort(40))["limits"]["used_pct"] == 40
    assert ck.auswerten(_antwort(0, fehler="401 Unauthorized: token revoked"))["state"] == "unauthorized"
    assert ck.auswerten("")["state"] == "probe_error"
    assert ck.auswerten(json.dumps({"id": 2, "result": {"account": None}}) + "\n"
                        + json.dumps({"id": 3, "result": {}}))["state"] == "logged_out"


def _stand(**pct):
    return {k: {"state": "ok", "limits": {"used_pct": v, "resets_at": 1, "erreicht": False}} for k, v in pct.items()}


@pytest.mark.parametrize("stand,ziel", [
    (_stand(apopovic=80, tsaier=10), None),            # nicht fast voll: bleibt
    (_stand(apopovic=96, tsaier=40), "tsaier"),        # fast voll, 56 Punkte Vorsprung
    (_stand(apopovic=96, tsaier=80), None),            # 95-99: nur 16 Punkte Vorsprung
    (_stand(apopovic=98, tsaier=95), None),            # 95-99: Ziel selbst fast voll
    ({**_stand(apopovic=99.5), "tsaier": {"state": "unauthorized", "limits": None}}, None),
    # Alex 20.11. „wechsel auf t saier bei 99%“ (Regel wie Login):
    (_stand(apopovic=99, tsaier=82), "tsaier"),        # ab 99 % ohne Vorsprung
    (_stand(apopovic=100, tsaier=97), "tsaier"),       # Ziel darf 95-98 sein
    (_stand(apopovic=99, tsaier=99), None),            # Ziel selbst ab 99 %
    (_stand(apopovic=99.2, tsaier=98.9), "tsaier"),    # weniger verbraucht, unter 99
    ({"apopovic": {"state": "ok", "limits": {"used_pct": 90, "resets_at": 1, "erreicht": True}},
      **_stand(tsaier=97)}, None),                      # erreicht, aber Ziel nicht weniger verbraucht
    ({"apopovic": {"state": "ok", "limits": {"used_pct": 90, "resets_at": 1, "erreicht": True}},
      **_stand(tsaier=82)}, "tsaier"),
])
def test_regel_wie_codex_konto_wechsel(stand, ziel):
    assert ck.entscheide("apopovic", stand) == ziel


def _lauf_mit(pcts, fehler=None):
    def lauf(cmd, **k):
        konto = cmd[-2]
        assert cmd[-1] == "limits" and cmd[4].endswith("/codex-als-dienst")
        f = (fehler or {}).get(konto)
        return SimpleNamespace(returncode=0, stdout=_antwort(pcts.get(konto, 0), fehler=f), stderr="")
    return lauf


def test_runde_wechselt_und_meldet(konto):
    gemeldet = []
    erg = ck.runde(melden=gemeldet.append, lauf=_lauf_mit({"apopovic": 97, "tsaier": 30}))
    assert erg == {"aktion": "gewechselt", "von": "apopovic", "nach": "tsaier"}
    assert ck.aktiv() == "tsaier"
    u = ck.uebersicht()
    assert u["aktiv"] == "tsaier" and u["wechsel"][-1]["von_pct"] == 97 and u["wechsel"][-1]["nach_pct"] == 30
    assert [e["art"] for e in gemeldet] == ["wechsel"]
    assert "Wechsel apopovic → tsaier" in gemeldet[0]["text"]
    assert {k["name"] for k in u["konten"]} == {"apopovic", "tsaier"}
    assert "/" not in json.dumps(u["konten"])           # keine Pfade


def test_runde_haelt_den_mindestabstand(konto):
    ck.runde(lauf=_lauf_mit({"apopovic": 97, "tsaier": 30}))
    erg = ck.runde(lauf=_lauf_mit({"apopovic": 10, "tsaier": 99}))
    assert erg["aktion"] == "abstand" and ck.aktiv() == "tsaier"


def test_kein_ziel_einmal_je_zustand_mit_ehrlichem_grund(konto):
    gemeldet = []
    for _ in range(3):
        erg = ck.runde(melden=gemeldet.append, lauf=_lauf_mit({"apopovic": 97, "tsaier": 85}))
    assert erg["aktion"] == "kein_ziel"
    assert [e["art"] for e in gemeldet] == ["kein_ziel"]
    assert "nur 12 Punkte" in gemeldet[0]["text"] and "spaetestens wenn apopovic 99 % erreicht" in gemeldet[0]["text"]
    ck.runde(melden=gemeldet.append, lauf=_lauf_mit({"apopovic": 50, "tsaier": 85}))   # wieder Luft
    ck.runde(melden=gemeldet.append, lauf=_lauf_mit({"apopovic": 97, "tsaier": 85}))
    assert [e["art"] for e in gemeldet] == ["kein_ziel", "kein_ziel"]


def test_abgelehnte_anmeldung_sofort_und_einmal(konto):
    gemeldet = []
    lauf = _lauf_mit({"apopovic": 10}, fehler={"tsaier": "invalid_grant"})
    ck.runde(melden=gemeldet.append, lauf=lauf)
    ck.runde(melden=gemeldet.append, lauf=lauf)
    assert [(e["art"], e["konto"]) for e in gemeldet] == [("anmeldung_abgelehnt", "tsaier")]
    ck.runde(melden=gemeldet.append, lauf=_lauf_mit({"apopovic": 10, "tsaier": 5}))
    ck.runde(melden=gemeldet.append, lauf=lauf)
    assert len(gemeldet) == 2


def test_runde_laeuft_nur_einmal_gleichzeitig(konto):
    import fcntl
    with open(str(konto.zustand) + ".lock", "a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        assert ck.runde(lauf=_lauf_mit({}))["aktion"] == "laeuft_schon"


def test_kontingentfehler_erkannt():
    assert ck.ist_kontingentfehler("ERROR: You've hit your usage limit. Try again at 3pm")
    assert ck.ist_kontingentfehler('{"type":"error","message":"usage_limit_reached"}')
    assert not ck.ist_kontingentfehler("Model not found")


# --- Endpunkt ----------------------------------------------------------------

def _claims(monkeypatch, claims):
    from ai.services import jwt_pruefung
    monkeypatch.setattr(jwt_pruefung, "pruefen", lambda a: claims)


@pytest.mark.asyncio
async def test_endpunkt_fuer_login(konto, monkeypatch):
    from ai.routes import codex_konten_routes as r
    _claims(monkeypatch, {"type": "agent", "sub": "agent:Login"})
    res = await r.konten_stand(authorization="Bearer x")
    body = json.loads(res.body)
    assert body["aktiv"] == "apopovic" and "regel" in body


@pytest.mark.asyncio
@pytest.mark.parametrize("claims", [None, {"type": "agent", "sub": "agent:Storage"},
                                    {"type": "service", "sub": "agent:Login"}])
async def test_endpunkt_sonst_zu(konto, monkeypatch, claims):
    from ai.routes import codex_konten_routes as r
    from ai.routes import verbrauch_routes
    _claims(monkeypatch, claims)

    async def kein_admin(a):
        return {"agentos_role": "user"}
    monkeypatch.setattr(verbrauch_routes, "_profil", kein_admin)
    with pytest.raises(HTTPException) as e:
        await r.konten_stand(authorization="Bearer x" if claims else None)
    assert e.value.status_code in (401, 403)


def test_health_im_kontomodus_aus_der_messung(konto):
    """Vorher las /health $CODEX_HOME/auth.json; im Kontomodus liegt dort keine
    Datei, /health meldete faelschlich 'absent'."""
    import main
    s = main._codex_auth_status()
    assert s["konto_modus"] and s["aktiv"] == "apopovic" and s["ok"] is None
    ck.runde(lauf=_lauf_mit({"apopovic": 10, "tsaier": 10}))
    assert main._codex_auth_status()["ok"] is True


def _wrapper_mit_attrappe(tmp_path, antwort_verzoegerung="0"):
    """limits gegen einen falschen app-server: antwortet auf id 2 und 3 und
    schlaeft danach 30 s, wie der echte, der auf stdin wartet."""
    p, konten = _wrapper(tmp_path)
    fake = tmp_path / "fake-app-server"
    fake.write_text(f"""#!/bin/bash
[[ "$1" == app-server ]] || exit 9
while IFS= read -r z; do
  echo "EINGABE $z" >> {tmp_path}/eingaben
  case "$z" in
    *'"id":2'*) sleep {antwort_verzoegerung}; echo '{{"id":2,"result":{{"account":{{"email":"x@y"}}}}}}' ;;
    *'"id":3'*) echo '{{"id":3,"result":{{"rateLimits":{{"primary":{{"usedPercent":42}}}}}}}}'; sleep 30 ;;
  esac
done
""")
    fake.chmod(0o755)
    s = p.read_text().replace("NODE=/usr/bin/node", "NODE=/bin/bash").replace(
        "BIN=/usr/local/lib/api-ai/codex/bin/codex.js", f"BIN={fake}")
    p.write_text(s)
    return p


def test_limits_endet_sobald_beide_antworten_da_sind(tmp_path):
    import time
    p = _wrapper_mit_attrappe(tmp_path)
    t0 = time.time()
    r = _lauf(p, "apopovic", "limits", stdin='{"id":9,"method":"command/exec"}\n')
    dauer = time.time() - t0
    assert r.returncode == 0, r.stderr
    m = ck.auswerten(r.stdout)
    assert m["state"] == "ok" and m["limits"]["used_pct"] == 42
    assert dauer < 5, dauer                                    # vorher immer 20 s
    eingaben = (tmp_path / "eingaben").read_text()
    assert "command/exec" not in eingaben                      # stdin des Aufrufers kommt nie an
