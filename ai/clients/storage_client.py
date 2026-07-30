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


# Storage API Configuration
STORAGE_API_URL = os.getenv("STORAGE_API_URL", "https://api-storage.arkturian.com")

# Legacy shim: the storage key literal used to be copy-pasted into 12 call
# sites across routes/ and services/, which is why rotating it had no effect
# (#514) — the env var could be changed all day while every module kept
# falling back to the baked-in string. Single source of truth now; rotating
# means setting STORAGE_API_KEY, and dropping the fallback is a one-line
# change here instead of a 12-file sweep.
_STORAGE_KEY_LEGACY_FALLBACK = "Inetpass1"
_warned_fallback = False


def storage_api_key() -> str:
    """Return the storage-api key. Prefers STORAGE_API_KEY, then API_KEY.

    Falls back to the historical literal so nothing breaks where the env
    var is not set yet (the dev .env has it empty) — but logs a warning
    once, so the fallback stops being invisible. Remove the fallback
    together with the rotation.
    """
    global _warned_fallback
    key = os.getenv("STORAGE_API_KEY") or os.getenv("API_KEY")
    if key:
        return key
    if not _warned_fallback:
        _warned_fallback = True
        logging.getLogger(__name__).warning(
            "STORAGE_API_KEY/API_KEY unset — using the legacy hardcoded "
            "fallback. Key rotation will NOT take effect until the env var "
            "is set on this host (#514)."
        )
    return _STORAGE_KEY_LEGACY_FALLBACK


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

    if collection_id:
        form_data["collection_id"] = collection_id
    if link_id:
        form_data["link_id"] = link_id
    if owner_email:
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
            headers={"X-API-KEY": STORAGE_API_KEY}
        )
        response.raise_for_status()

        # Parse response
        result = response.json()

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
