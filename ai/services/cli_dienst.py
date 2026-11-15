"""CLI-Aufrufe aus dem Dienstbenutzer apiai (#2937, Beitrag 6150).

Solange api-ai als root laeuft, ist nichts davon aktiv: Ohne
API_AI_CLI_WRAPPER_DIR bleibt jeder Pfad wie bisher (sudo -u <user> env ...,
chown an den CLI-Benutzer).

Mit API_AI_CLI_WRAPPER_DIR (Cutover auf User=apiai):
- codex und agy starten nur ueber die root-eigenen Wrapper, die sudoers
  allein erlaubt. Die Wrapper setzen die Umgebung selbst (env -i); api-ai
  reicht keine Variable mehr durch.
- apiai darf nicht chown-en. Bilder und Arbeitsordner kommen deshalb in
  setgid-Verzeichnisse der Gruppe apiai-cli (eingang/, arbeit/), lesbar ueber
  die Gruppe (0640 / 0750), statt dem CLI-Benutzer zu gehoeren.
- Was nur alex lesen kann (config.toml), liefert der codex-Wrapper.
"""
from __future__ import annotations

import os
import subprocess
import tomllib
from typing import List, Optional, Tuple

CODEX_WRAPPER = "codex-als-alex"
AGY_WRAPPER = "agy-als-dienst"


def wrapper_dir() -> Optional[str]:
    d = (os.getenv("API_AI_CLI_WRAPPER_DIR") or "").strip()
    return d or None


def aktiv() -> bool:
    return wrapper_dir() is not None


def eingang() -> str:
    return (os.getenv("API_AI_CLI_EINGANG") or "/var/lib/api-ai-cli/eingang").rstrip("/")


def arbeit() -> str:
    return (os.getenv("API_AI_CLI_ARBEIT") or "/var/lib/api-ai-cli/arbeit").rstrip("/")


def codex_text_home() -> str:
    return (os.getenv("CODEX_TEXT_HOME") or "/var/lib/api-ai-cli/codex-text").rstrip("/")


def codex_befehl(user: str, cmd: List[str], env: dict) -> List[str]:
    """sudo -n -u <user> <wrapper> haupt|text <codex-Argumente>. Welches
    CODEX_HOME gilt, entscheidet der Wrapper aus einer festen Liste."""
    art = "text" if (env.get("CODEX_HOME") or "").rstrip("/") == codex_text_home() else "haupt"
    return ["sudo", "-n", "-u", user, os.path.join(wrapper_dir(), CODEX_WRAPPER), art] + list(cmd[1:])


def agy_befehl(user: str, cmd: List[str]) -> List[str]:
    return ["sudo", "-n", "-u", user, os.path.join(wrapper_dir(), AGY_WRAPPER)] + list(cmd[1:])


def gruppenlesbar_ablegen(ziel: str) -> None:
    """Statt chown: Die Datei liegt im setgid-Eingang (Gruppe apiai-cli),
    0640 macht sie fuer den CLI-Benutzer ueber die Gruppe lesbar."""
    os.chmod(ziel, 0o640)


def arbeitsordner(prefix: str) -> str:
    """Wegwerf-Arbeitsordner unter arbeit/, den der CLI-Benutzer betreten
    darf (Gruppe ueber setgid, 0750). Der Aufrufer raeumt auf."""
    import tempfile
    os.makedirs(arbeit(), exist_ok=True)
    d = tempfile.mkdtemp(prefix=prefix, dir=arbeit())
    os.chmod(d, 0o750)
    return d


def codex_konfig(user: str) -> Tuple[Optional[str], Optional[str]]:
    """model und model_reasoning_effort aus alex' config.toml ueber den
    Wrapper (`konfig` gibt nur diese beiden Zeilen aus)."""
    try:
        r = subprocess.run(["sudo", "-n", "-u", user, os.path.join(wrapper_dir(), CODEX_WRAPPER), "konfig"],
                           capture_output=True, text=True, timeout=10)
        cfg = tomllib.loads(r.stdout or "") if r.returncode == 0 else {}
    except Exception:
        return None, None
    m, e = cfg.get("model"), cfg.get("model_reasoning_effort")
    return (m if isinstance(m, str) and m else None), (e if isinstance(e, str) and e else None)


class arbeitsordner_kontext:
    """with-Form von arbeitsordner(): legt an, raeumt danach weg."""

    def __init__(self, prefix: str):
        self.prefix = prefix
        self.pfad: Optional[str] = None

    def __enter__(self) -> str:
        self.pfad = arbeitsordner(self.prefix)
        return self.pfad

    def __exit__(self, *exc) -> None:
        import shutil
        if self.pfad:
            shutil.rmtree(self.pfad, ignore_errors=True)
