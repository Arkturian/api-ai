"""Identitaets-Strings fuer Logzeilen kuerzen — ohne sie zu verwechseln.

Warum es dieses Modul gibt
--------------------------
Neun Logstellen kuerzten die Grant-Identitaet mit ``sub[:8]``. Das war
richtig, solange jedes ``sub`` eine UUID war: acht Hexzeichen am Anfang
unterscheiden zuverlaessig.

Seit AuthApi Service-Principals ausstellt (``fe3e55b``) traegt ``sub``
die Form ``agent:<name>``. Das Unterscheidende steht damit am **Ende**,
nicht am Anfang — jeder Agent-Principal kuerzt auf ``agent:pr``:

    agent:productfinder-internal-demo  ->  agent:pr
    agent:productfinder-batch          ->  agent:pr

Mit genau einem Principal faellt das nicht auf. Es faellt beim zweiten
auf, und zwar im Vorfall: Die Logzeile, mit der man klaeren will, wer
das Budget verbraucht hat, sagt dann fuer beide dasselbe. Ein Log, das
zwei Verursacher gleich benennt, ist schlimmer als eines, das gar
nichts sagt — es beantwortet die Frage falsch, statt sie offen zu
lassen.

Was hier NICHT passiert
-----------------------
Das ist eine Kuerzung fuer Menschenaugen, keine Anonymisierung. Der
Agent-Name steht danach im Klartext im Log — er ist auch keine PII,
sondern eine Dienstkennung. Die UUID-Haelfte bleibt gekuerzt, weil sie
eine Person bezeichnet.
"""

from __future__ import annotations

from typing import Optional

# Praefixe, hinter denen eine Dienstkennung steht statt einer UUID.
# AuthApi setzt sie in `_issue_grant_agent` als f"agent:{agent_name}".
_DIENST_PRAEFIXE = ("agent:", "service:")


def kurz_id(roh: Optional[str], laenge: int = 8) -> str:
    """Kuerze eine Grant-Identitaet fuer eine Logzeile.

    * ``agent:<name>`` -> vollstaendig. Der Name IST die Kennung, und
      er ist kurz genug; ihn zu kuerzen wirft genau das weg, was ihn
      unterscheidet.
    * alles andere (UUID, Opaque-Sub) -> die ersten ``laenge`` Zeichen,
      wie bisher. Diese Strings bezeichnen Personen; sie bleiben
      gekuerzt.
    * ``None``/leer -> ``"-"``, damit eine Logzeile nie ``None`` sagt
      und nie mit einem echten Wert zu verwechseln ist.
    """
    if not roh:
        return "-"
    text = str(roh)
    if text.startswith(_DIENST_PRAEFIXE):
        return text
    return text[:laenge]
