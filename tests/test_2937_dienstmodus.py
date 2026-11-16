"""#2937: api-ai als Dienstbenutzer apiai (Beitrag 6150).

Mit API_AI_CLI_WRAPPER_DIR laufen codex und agy nur ueber die root-eigenen
Wrapper, und nichts wird mehr ge-chown-t. Ohne die Variable bleibt alles wie
bisher (Dienst als root). Die Wrapper selbst stehen in deploy/apiai/ und
werden hier gegen echte und feindliche Argumentlisten geprueft.
"""
import os
import pwd
import stat
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from ai.routes import text_ai_routes as t
from ai.services import cli_dienst, cli_katalog

WRAPPER = "/usr/local/libexec/api-ai"
REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def dienst(monkeypatch, tmp_path):
    ein, arb, text = tmp_path / "eingang", tmp_path / "arbeit", tmp_path / "codex-text"
    for d in (ein, arb, text):
        d.mkdir()
    monkeypatch.setenv("API_AI_CLI_WRAPPER_DIR", WRAPPER)
    monkeypatch.setenv("API_AI_CLI_EINGANG", str(ein))
    monkeypatch.setenv("API_AI_CLI_ARBEIT", str(arb))
    monkeypatch.setenv("CODEX_TEXT_HOME", str(text))
    monkeypatch.setenv("CODEX_RUN_AS_USER", "alex")

    def kein_chown(*a, **k):
        raise AssertionError("im Dienstmodus darf nichts ge-chown-t werden")
    monkeypatch.setattr(os, "chown", kein_chown)
    monkeypatch.setattr(os, "lchown", kein_chown)
    return SimpleNamespace(eingang=ein, arbeit=arb, text=text)


def _fremder_benutzer():
    for name in ("alex", "nobody", "root"):
        try:
            pw = pwd.getpwnam(name)
        except KeyError:
            continue
        if pw.pw_uid != os.geteuid():
            return name
    pytest.skip("kein zweiter Benutzer")


# --- codex ------------------------------------------------------------------

def test_codex_nur_ueber_wrapper_ohne_umgebung(dienst):
    cmd = t.codex_als_benutzer(["codex", "exec", "--json", "-"],
                               {"CODEX_HOME": "/home/alex/.codex", "PATH": "/evil", "HOME": "/x"})
    assert cmd == ["sudo", "-n", "-u", "alex", f"{WRAPPER}/codex-als-alex", "haupt", "exec", "--json", "-"]
    assert "env" not in cmd and not any(a.startswith(("PATH=", "CODEX_HOME=")) for a in cmd)


def test_codex_text_home_waehlt_wrapper_art_text(dienst):
    cmd = t.codex_als_benutzer(["codex", "debug", "models"], {"CODEX_HOME": str(dienst.text)})
    assert cmd[5] == "text"


def test_ohne_dienstmodus_bleibt_der_alte_weg(monkeypatch):
    monkeypatch.delenv("API_AI_CLI_WRAPPER_DIR", raising=False)
    monkeypatch.setenv("CODEX_RUN_AS_USER", "alex")
    cmd = t.codex_als_benutzer(["codex", "exec", "-"], {"CODEX_HOME": "/h", "PATH": "/usr/bin"})
    assert cmd[:6] == ["sudo", "-n", "-u", "alex", "-H", "env"]


def test_text_home_ohne_dateioperationen(dienst, monkeypatch):
    monkeypatch.setattr(os, "symlink", lambda *a: (_ for _ in ()).throw(AssertionError("symlink")))
    assert t._codex_text_home("/home/alex/.codex") == str(dienst.text)


def test_vision_kopie_in_den_eingang_gruppenlesbar(dienst, monkeypatch, tmp_path):
    monkeypatch.setenv("CODEX_RUN_AS_USER", _fremder_benutzer())
    wurzel = tmp_path / "disk"
    wurzel.mkdir()
    bild = wurzel / "a.jpg"
    bild.write_bytes(b"JPG")
    monkeypatch.setenv("VISION_LOKALE_WURZELN", str(wurzel))
    neu, kopien = t.bilder_fuer_codex_benutzer([str(bild)])
    assert len(kopien) == 1 and kopien[0].startswith(str(dienst.eingang))
    assert stat.S_IMODE(os.stat(kopien[0]).st_mode) == 0o640


def test_config_effort_ueber_wrapper(dienst, monkeypatch):
    rufe = []

    def lauf(cmd, **k):
        rufe.append(cmd)
        return SimpleNamespace(returncode=0, stdout='model = "gpt-6.1-terra"\nmodel_reasoning_effort = "High"\n')
    monkeypatch.setattr(cli_dienst.subprocess, "run", lauf)
    assert t._codex_config_effort() == "high"
    assert rufe == [["sudo", "-n", "-u", "alex", f"{WRAPPER}/codex-als-alex", "konfig"]]


def test_katalog_config_bei_gesperrtem_home_ueber_wrapper(dienst, monkeypatch, tmp_path):
    gesperrt = tmp_path / "zu"
    gesperrt.mkdir()

    def lies(self, *a, **k):
        raise PermissionError(13, "Permission denied")
    monkeypatch.setattr(Path, "read_text", lies)
    monkeypatch.setattr(cli_dienst.subprocess, "run", lambda cmd, **k: SimpleNamespace(
        returncode=0, stdout='model = "gpt-6.1-terra"\nmodel_reasoning_effort = "medium"\n'))
    assert cli_katalog.codex_config(str(gesperrt)) == ("gpt-6.1-terra", "medium")


def test_arbeitsordner_fuer_alex_betretbar(dienst):
    with cli_dienst.arbeitsordner_kontext("codex-imagegen-") as d:
        assert d.startswith(str(dienst.arbeit))
        assert stat.S_IMODE(os.stat(d).st_mode) == 0o750
    assert not os.path.exists(d)


# --- agy --------------------------------------------------------------------

def test_agy_nur_ueber_wrapper(dienst):
    cmd = t._agy_als_benutzer(["agy", "--output-format", "json", "--sandbox"], "apiai-agy", "/var/lib/apiai-agy",
                              {"PATH": "/evil"})
    assert cmd == ["sudo", "-n", "-u", "apiai-agy", f"{WRAPPER}/agy-als-dienst",
                   "--output-format", "json", "--sandbox"]


def test_agy_arbeitsbereich_und_bilder_ohne_chown(dienst, monkeypatch, tmp_path):
    user = _fremder_benutzer()
    assert t._agy_arbeitsbereich("/var/lib/apiai-agy", user) == str(dienst.arbeit)
    quelle = tmp_path / "x.png"
    quelle.write_bytes(b"PNG")
    [(url, ziel)] = t._agy_bilder_uebergeben([("https://s/1", str(quelle))], "/var/lib/apiai-agy", user)
    assert ziel.startswith(str(dienst.eingang)) and not quelle.exists()
    assert stat.S_IMODE(os.stat(ziel).st_mode) == 0o640


# --- Wrapper-Skripte -----------------------------------------------------------

def _wrapper(tmp_path, name):
    """Kopie des Wrappers, deren exec nur ausgibt, was gestartet wuerde."""
    s = (REPO / "deploy" / "apiai" / name).read_text()
    s = s.replace("exec env -i", "echo START env -i").replace("cd /var/lib/apiai-agy/arbeit", ":")
    s = s.replace('[[ "$CH" == "$HAUPT" ]] || text_home_heilen', ":")
    p = tmp_path / name
    p.write_text(s)
    p.chmod(0o755)
    return p


def _lauf(p, *args, stdin="hallo"):
    return subprocess.run([str(p), *args], input=stdin, capture_output=True, text=True)


A = "/var/www/api-ai.arkturian.com/ai/data/codex_text_anweisung.md"
E = "/var/lib/api-ai-cli/eingang"
ECHTE_CODEX = [
    ["text", "exec", "--json", "--skip-git-repo-check", "-c", "features.shell_tool=false",
     "-c", "features.unified_exec=false", "-c", "features.code_mode_host=false", "-c", "features.apps=false",
     "-c", "features.plugins=false", "--ephemeral", "-c", f"model_instructions_file={A}",
     "-c", "include_environment_context=false", "-c", "include_permissions_instructions=false",
     "-c", "include_apps_instructions=false", "--model", "gpt-6.1-terra", "-c", "model_reasoning_effort=high",
     "-i", f"{E}/aiimg_1.jpg", "--", "-"],
    ["text", "exec", "--json", "--skip-git-repo-check", "-s", "read-only", "-C",
     "/var/lib/api-ai-cli/arbeit/codex-imagegen-x", "-m", "gpt-6", "-i", f"{E}/r1.png", "--", "-"],
    ["haupt", "exec", "--json", "--skip-git-repo-check", "-"],
    ["text", "debug", "models"],
    ["haupt", "--version"],
]
FEINDLICH_CODEX = [
    ["haupt", "exec", "--dangerously-bypass-approvals-and-sandbox", "-"],
    ["haupt", "exec", "-c", "sandbox_mode=danger-full-access", "-"],
    ["haupt", "exec", "-c", "features.shell_tool=true", "-"],
    ["haupt", "exec", "-s", "workspace-write", "-"],
    ["haupt", "exec", "-i", "/home/alex/.ssh/id_ed25519", "--", "-"],
    ["haupt", "exec", "-i", f"{E}/../../../../home/alex/.ssh/id", "--", "-"],
    ["haupt", "exec", "-C", "/", "-"],
    ["haupt", "exec", "-p", "evil", "-"],
    ["haupt", "exec", "--model", "-x", "-"],
    ["haupt", "exec", "-c", "model_reasoning_effort=high;x", "-"],
    ["haupt", "exec", "-", "extra"],
    ["haupt", "exec", "--json", "mach was"],
    ["haupt", "mcp", "list"],
    ["/tmp", "exec", "-"],
    ["haupt", "exec", "--add-dir", "/", "-"],
    ["haupt", "exec", "--", "--yolo"],
]


@pytest.mark.parametrize("args", ECHTE_CODEX)
def test_codex_wrapper_laesst_echte_aufrufe_durch(tmp_path, args):
    r = _lauf(_wrapper(tmp_path, "codex-als-alex"), *args)
    assert r.returncode == 0 and r.stdout.startswith("START env -i HOME=/home/alex"), r.stderr


@pytest.mark.parametrize("args", FEINDLICH_CODEX)
def test_codex_wrapper_weist_angriffe_ab(tmp_path, args):
    r = _lauf(_wrapper(tmp_path, "codex-als-alex"), *args)
    assert r.returncode == 64 and "START" not in r.stdout


def test_codex_wrapper_erzwingt_read_only(tmp_path):
    r = _lauf(_wrapper(tmp_path, "codex-als-alex"), "haupt", "exec", "--json", "-")
    assert " -s read-only --json -" in r.stdout


def _gesperrt_im_wrapper():
    import re
    s = (REPO / "deploy" / "apiai" / "codex-als-alex").read_text()
    return re.search(r"GESPERRT=\(([^)]*)\)", s).group(1).split()


@pytest.mark.parametrize("modus", ["haupt", "text"])
def test_3165_wrapper_erzwingt_werkzeuge_aus_in_jedem_modus(tmp_path, modus):
    """Vorher erlaubte der Wrapper features.*=false nur; im Modus haupt las
    codex als alex per Shell jede Datei von alex (Kanarienprobe 16.11.)."""
    r = _lauf(_wrapper(tmp_path, "codex-als-alex"), modus, "exec", "--json", "-")
    assert r.returncode == 0, r.stderr
    kopf = r.stdout.split(" exec ", 1)[1]
    for f in ("shell_tool", "unified_exec", "plugins", "apps", "browser_use", "view_image",
              "computer_use"):
        assert f"-c features.{f}=false" in kopf
    # image_gen laeuft ueber den Code-Modus: nicht erzwungen aus
    assert "features.code_mode_host=false" not in kopf


def test_3165_wrapper_und_code_sperren_dasselbe():
    from ai.routes.text_ai_routes import CODEX_GESPERRT
    assert set(_gesperrt_im_wrapper()) == set(CODEX_GESPERRT)


@pytest.mark.parametrize("f", ["browser_use", "view_image", "plugins", "computer_use"])
def test_3165_wrapper_laesst_einschalten_nicht_zu(tmp_path, f):
    r = _lauf(_wrapper(tmp_path, "codex-als-alex"), "haupt", "exec", "-c", f"features.{f}=true", "-")
    assert r.returncode == 64 and "START" not in r.stdout


def test_3165_code_sperren_passieren_den_wrapper(tmp_path):
    from ai.routes.text_ai_routes import CODEX_OHNE_WERKZEUGE
    r = _lauf(_wrapper(tmp_path, "codex-als-alex"), "text", "exec", "--json", *CODEX_OHNE_WERKZEUGE, "-")
    assert r.returncode == 0, r.stderr


def test_agy_wrapper(tmp_path):
    w = _wrapper(tmp_path, "agy-als-dienst")
    ok = _lauf(w, "--output-format", "json", "--sandbox", "--model", "gemini-4-carbon-medium", "--effort", "high")
    assert ok.returncode == 0 and "HOME=/var/lib/apiai-agy" in ok.stdout
    for boese in (["--output-format", "json", "-p", "text"], ["--output-format", "json"],
                  ["--sandbox", "--yolo"], ["--sandbox", "--dangerously-skip-permissions"],
                  ["--sandbox", "--approval-mode", "yolo"], ["--sandbox", "--model", "-x"]):
        r = _lauf(w, *boese)
        assert r.returncode == 64 and "START" not in r.stdout, boese


def test_sudoers_nur_zwei_programme_ohne_setenv():
    zeilen = [z for z in (REPO / "deploy" / "apiai" / "sudoers-api-ai").read_text().splitlines()
              if z.strip() and not z.startswith("#")]
    regeln = [z for z in zeilen if z.startswith("apiai ALL=")]
    assert len(regeln) == 2 and all("NOSETENV:" in z and " ALL\n" not in z for z in regeln)
    assert {z.split()[-1] for z in regeln} == {f"{WRAPPER}/codex-als-alex", f"{WRAPPER}/agy-als-dienst"}
    assert any("!setenv" in z and "env_reset" in z for z in zeilen if z.startswith("Defaults:apiai"))


def _heilen(tmp_path, ziel_schreibbar=True):
    """text_home_heilen aus dem echten Wrapper gegen ein Abbild von alex' Home:
    haupt/auth.json -> konten/apopovic/auth.json, Text-Home mit einer frisch
    von codex erneuerten Datei statt des Links."""
    s = (REPO / "deploy" / "apiai" / "codex-als-alex").read_text()
    a = s.index("text_home_heilen() {")
    fn = s[a:s.index("\n}\n", a) + 3]
    konten = tmp_path / "konten" / "apopovic"
    konten.mkdir(parents=True)
    echt = konten / "auth.json"
    echt.write_text("alt")
    os.utime(echt, (1, 1))
    haupt = tmp_path / "haupt"
    haupt.mkdir()
    (haupt / "auth.json").symlink_to(echt)
    ch = tmp_path / "text"
    ch.mkdir()
    link = ch / "auth.json"
    link.write_text("frisch")
    if not ziel_schreibbar:
        konten.chmod(0o500)  # wie EROFS unter ProtectHome=read-only
    try:
        r = subprocess.run(["bash", "-c", f"set -euo pipefail\nHAUPT={haupt}\nCH={ch}\n{fn}\ntext_home_heilen\necho OK"],
                           capture_output=True, text=True)
    finally:
        konten.chmod(0o700)
    return r, echt, link


def test_heilen_schreibt_frische_anmeldung_zurueck(tmp_path):
    r, echt, link = _heilen(tmp_path)
    assert r.stdout.strip() == "OK", r.stderr
    assert echt.read_text() == "frisch" and link.is_symlink()


@pytest.mark.skipif(os.geteuid() == 0, reason="root schreibt trotz 0500")
def test_heilen_loescht_nie_die_einzige_frische_anmeldung(tmp_path):
    """Cloud 16.11.: cp scheiterte (EROFS), rm loeschte danach den Text-Home-
    Stand, das Haupt-Home behielt das verbrauchte Refresh-Token."""
    r, echt, link = _heilen(tmp_path, ziel_schreibbar=False)
    assert r.stdout.strip() == "OK", r.stderr
    assert not link.is_symlink() and link.read_text() == "frisch"
    assert echt.read_text() == "alt"
    assert "nicht" in r.stderr and "zurueckgeschrieben" in r.stderr
