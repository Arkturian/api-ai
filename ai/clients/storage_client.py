"""
Storage API HTTP Client

Provides functions to upload files to the Storage API
and manage storage objects without direct database access.
"""

import httpx
import mimetypes
import logging
import os
from typing import Optional, Dict, Any
from pydantic import BaseModel
from ai.services import aufruf as _aufruf  # noqa: E402  #4132


class StorageObject(BaseModel):
    """Storage Object Response Model"""
    id: int
    object_key: str
    original_filename: str
    mime_type: Optional[str] = None
    file_size: Optional[int] = None
    file_url: str
    thumbnail_url: Optional[str] = None
    webview_url: Optional[str] = None
    context: Optional[str] = None
    is_public: bool = False
    collection_id: Optional[str] = None
    link_id: Optional[str] = None
    ai_title: Optional[str] = None
    ai_subtitle: Optional[str] = None
    ai_tags: Optional[list] = None
    ai_safety_rating: Optional[str] = None
    # Scope, der bei DIESEM Upload wirklich gebunden wurde (storage-api seit
    # 06f8e7f, 11.10.); None = nicht gebunden. Bei einem spaeteren GET bleibt
    # das Feld None, es beschreibt nur den Upload-Aufruf selbst.
    grant_scope: Optional[str] = None


class GrantNichtGebunden(Exception):
    """Der Storage hat das Objekt gespeichert, aber NICHT an den verlangten
    Scope gebunden. Traegt die Objekt-ID, damit der Aufrufer sie nennen kann."""

    def __init__(self, object_id, verlangt, gebunden):
        self.object_id, self.verlangt, self.gebunden = object_id, verlangt, gebunden
        super().__init__(f"storage object {object_id}: grant_scope {verlangt!r} verlangt, {gebunden!r} gebunden")


# Storage API Configuration
STORAGE_API_URL = os.getenv("STORAGE_API_URL", "https://api-storage.arkturian.com")

# #1960: Der fest eingebaute Rueckfall-Schluessel ist entfernt. Jeder Host
# bekommt seinen eigenen Storage-Schluessel ueber STORAGE_API_KEY (Datei,
# EnvironmentFile). Fehlt er, gibt es keinen Ersatz: Storage antwortet 401,
# und das Log sagt einmal laut, warum. Frueher (#514) verdeckte der Rueckfall
# jede Rotation.
_warned_missing = False


def storage_api_key() -> str:
    """Return the storage-api key from STORAGE_API_KEY, else API_KEY. No literal fallback."""
    global _warned_missing
    key = (os.getenv("STORAGE_API_KEY") or os.getenv("API_KEY") or "").strip()
    if key:
        return key
    if not _warned_missing:
        _warned_missing = True
        logging.getLogger(__name__).error(
            "STORAGE_API_KEY ist nicht gesetzt: Aufrufe an storage-api laufen "
            "ohne Schluessel und scheitern (#1960)."
        )
    return ""


# Back-compat for modules importing the constant directly.
STORAGE_API_KEY = storage_api_key()


async def save_file_and_record(
    data: bytes,
    original_filename: str,
    context: Optional[str] = None,
    is_public: bool = True,
    collection_id: Optional[str] = None,
    link_id: Optional[str] = None,
    owner_user_id: Optional[int] = None,
    owner_email: Optional[str] = "apopovic.aut@gmail.com",  # Default from legacy
    analyze: bool = False,  # Skip AI analysis for AI-generated content
    skip_ai_safety: bool = True,  # Skip safety check for AI-generated content
    private: bool = False,  # vertraulich: /storage/media liefert nur an Eigentuemer/Admin (403 sonst)
    grant_scope: Optional[str] = None,  # bindet das Objekt beim Upload an einen Scope (media_grants)
    ttl_hours: Optional[int] = None,  # Storage loescht das Objekt nach N Stunden selbst
    reuse_existing: bool = False,  # AI-generated content is always a new work — storage's
    # filename-based reuse (its default!) silently OVERWRITES an earlier object when two
    # generations finish in the same second (timestamp filenames, Issue #512: two parallel
    # genimage calls landed on one storage id, one image lost).
) -> StorageObject:
    """
    Upload a file to the Storage API via HTTP.

    This replaces the direct database `save_file_and_record` function
    from the legacy API, enabling clean separation between services.

    Args:
        data: File bytes to upload
        original_filename: Original filename
        context: Context tag (e.g., "tts-generation", "dialog-audio")
        is_public: Whether the file is publicly accessible
        collection_id: Collection name/ID
        link_id: Link ID for related objects
        owner_user_id: Owner user ID (optional, owner_email takes precedence)
        owner_email: Owner email (defaults to apopovic.aut@gmail.com)
        analyze: Whether to run AI analysis (default False for AI-generated content)
        skip_ai_safety: Skip AI safety check (default True for AI-generated content)

    Returns:
        StorageObject with id, file_url, etc.

    Raises:
        httpx.HTTPStatusError: If the upload fails
    """

    # Prepare form data
    form_data = {
        "context": context or "ai-generated",
        "reuse_existing": str(reuse_existing).lower(),
        "is_public": str(is_public).lower(),
        "analyze": str(analyze).lower(),
        "skip_ai_safety": str(skip_ai_safety).lower(),
    }

    if private:
        # Storage-Vertrag (Issue #420): Formularfeld `private` setzt
        # metadata_json.private_media. Vertraulich heisst immer auch nicht
        # oeffentlich — ein privates, aber als oeffentlich gelistetes Objekt
        # waere ein Widerspruch, den der Aufrufer nicht bemerken wuerde.
        form_data["private"] = "true"
        form_data["is_public"] = "false"
    if grant_scope:
        form_data["grant_scope"] = grant_scope
    if ttl_hours:
        form_data["ttl_hours"] = str(int(ttl_hours))
    if collection_id:
        form_data["collection_id"] = collection_id
    if link_id:
        form_data["link_id"] = link_id
    # Mit grant_scope KEIN fremdes Besitzerkonto angeben. Der Storage bindet
    # beim Upload nur, wenn der Hochladende selbst Besitzer ist
    # (_bind_on_upload). Mit der owner_email-Vorgabe gehoerte das Objekt
    # einem anderen Benutzer als dem Schluessel: die Bindung wurde STILL
    # uebersprungen ("upload grant skipped ... owner mismatch"), der Upload
    # meldete trotzdem 200. Gemessen am 11.10. an Objekt 130172; ohne
    # owner_email entstand die Bindung (130174).
    if owner_email and not grant_scope:
        form_data["owner_email"] = owner_email

    # Detect MIME type from filename extension
    mime_type, _ = mimetypes.guess_type(original_filename)
    if not mime_type:
        mime_type = "application/octet-stream"

    # Prepare file
    files = {
        "file": (original_filename, data, mime_type)
    }

    # Upload to Storage API
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{STORAGE_API_URL}/storage/upload",
            files=files,
            data=form_data,
            headers={"X-API-KEY": STORAGE_API_KEY, **_aufruf.herkunft_kopf()}  # #4132
        )
        response.raise_for_status()

        # Parse response
        result = response.json()

        # Bindung am Ziel pruefen, nicht an der Naht: Der Storage antwortet
        # auch dann mit 200, wenn er die Bindung nicht angelegt hat. Am 11.10.
        # meldete /ai/genimage so einen Scope als gesetzt, der nie gebunden
        # war (Objekt 130172). Seit storage-api 06f8e7f nennt die Antwort den
        # wirklich gebundenen Scope. Stimmt er nicht, ist das ein Fehler —
        # das Objekt liegt dann privat und fuer den Besteller unlesbar da.
        if grant_scope and result.get("grant_scope") != grant_scope:
            raise GrantNichtGebunden(
                object_id=result.get("id"), verlangt=grant_scope, gebunden=result.get("grant_scope"))

        # Return as StorageObject
        return StorageObject(**result)


async def get_storage_object(object_id: int) -> StorageObject:
    """
    Fetch a storage object by ID.

    Args:
        object_id: Storage object ID

    Returns:
        StorageObject

    Raises:
        httpx.HTTPStatusError: If object not found or request fails
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(
            f"{STORAGE_API_URL}/storage/objects/{object_id}",
            headers={"X-API-KEY": STORAGE_API_KEY}
        )
        response.raise_for_status()

        result = response.json()
        return StorageObject(**result)
