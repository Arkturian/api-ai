"""Ereignisse der codex-Kontowahl an Login melden (#3165).

Vereinbart mit Login (20.11.): IACP nur bei Wechsel, bei „kein Ziel“ (einmal je
Zustand) und bei abgelehnter Anmeldung. Der Dienst braucht dafuer einen eigenen
Ausweis (`service:api-ai`, Recht nur zum Senden an Login), den es noch nicht
gibt; cloud-api muss ihn an /api/agents/send erst annehmen (Klaerung mit
Cloud-worker und AuthAPI). Bis dahin meldet diese Funktion nichts, und Login
liest die Ereignisse aus GET /ai/codex/konten.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def melden(ereignis: dict) -> None:
    logger.info("codex-konten: Ereignis %s (Meldung an Login noch ohne Ausweis, steht in /ai/codex/konten)",
                ereignis.get("art"))
