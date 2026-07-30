"""
Storage API HTTP Client

Provides functions to upload files to the Storage API
and manage storage objects without direct database access.
"""

import httpx
import mimetypes
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
STORAGE_API_KEY = os.getenv("API_KEY", "Inetpass1")  # Use the same API key


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
