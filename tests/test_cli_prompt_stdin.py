"""Prompt ueber stdin statt argv (#4914 q-936e78536864): argv ist auf
~128 KB je Argument begrenzt (MAX_ARG_STRLEN). Ein Spannenkontext oder ein
langes Dokument als Argument endet in "Argument list too long", bevor codex
ueberhaupt startet. Gegen den echten Fehler gehalten: 200 KB als argv
scheitern, 200 KB ueber stdin kommen an."""

import os
import subprocess

import pytest

from ai.routes import text_ai_routes as t


def test_runner_reicht_input_ueber_stdin_durch():
    r = t._run_cli_with_pgid(["cat"], env=dict(os.environ), timeout=10, input="hallo\nwelt")
    assert r.returncode == 0 and r.stdout == "hallo\nwelt"


def test_200kb_argv_scheitert_stdin_nicht():
    big = "x" * 200_000
    with pytest.raises(OSError):                       # E2BIG: Argument list too long
        subprocess.run(["true", big], capture_output=True)
    r = t._run_cli_with_pgid(["wc", "-c"], env=dict(os.environ), timeout=10, input=big)
    assert r.returncode == 0 and int(r.stdout.strip()) == 200_000


def test_ohne_input_bleibt_stdin_unangetastet():
    r = t._run_cli_with_pgid(["cat"], env=dict(os.environ), timeout=10)
    assert r.returncode == 0 and r.stdout == ""       # nichts haengt an einem offenen stdin
