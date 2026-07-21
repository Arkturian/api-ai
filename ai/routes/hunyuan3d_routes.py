"""
Hunyuan 3D Routes — Tencent Cloud Hunyuan: text or image → 3D model.

POST /ai/gen3d              — submit a Hunyuan-To-3D Pro job (text OR image);
                              optional sync wait + save result to storage
GET  /ai/gen3d/status/{id}  — poll a job, returns result file URLs when DONE

Upstream: `hunyuan.intl.tencentcloudapi.com` (service "hunyuan", version
2023-09-01), actions SubmitHunyuanTo3DProJob / QueryHunyuanTo3DProJob,
TC3-HMAC-SHA256 request signing (implemented inline — no SDK dependency).
Result URLs from Tencent expire (~24h), so results are persisted to
storage-api by default.

Billing: Tencent PAYG → default-deny `confirm_api_billing` gate (same
house rule + shared monthly cap as the other API-billed endpoints).
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import asyncio
import hashlib
import hmac
import json
import logging
import os
import time
import uuid

logger = logging.getLogger(__name__)
router = APIRouter()


def get_api_key():
    return "Inetpass1"


# ── TC3-HMAC-SHA256 signing ─────────────────────────────────────────────

def _tc3_headers(action: str, payload: bytes, host: str, service: str,
                 version: str, region: str, secret_id: str, secret_key: str) -> Dict[str, str]:
    """Build signed headers per Tencent Cloud TC3-HMAC-SHA256 spec."""
    ts = int(time.time())
    date = time.strftime("%Y-%m-%d", time.gmtime(ts))
    ct = "application/json; charset=utf-8"
    canonical = (
        "POST\n/\n\n"
        f"content-type:{ct}\nhost:{host}\nx-tc-action:{action.lower()}\n\n"
        "content-type;host;x-tc-action\n"
        + hashlib.sha256(payload).hexdigest()
    )
    scope = f"{date}/{service}/tc3_request"
    to_sign = (
        "TC3-HMAC-SHA256\n" + str(ts) + "\n" + scope + "\n"
        + hashlib.sha256(canonical.encode()).hexdigest()
    )

    def _hmac(key: bytes, msg: str) -> bytes:
        return hmac.new(key, msg.encode(), hashlib.sha256).digest()

    k_date = _hmac(("TC3" + secret_key).encode(), date)
    k_service = _hmac(k_date, service)
    k_signing = _hmac(k_service, "tc3_request")
    signature = hmac.new(k_signing, to_sign.encode(), hashlib.sha256).hexdigest()

    return {
        "Authorization": (
            f"TC3-HMAC-SHA256 Credential={secret_id}/{scope}, "
            f"SignedHeaders=content-type;host;x-tc-action, Signature={signature}"
        ),
        "Content-Type": ct,
        "Host": host,
        "X-TC-Action": action,
        "X-TC-Timestamp": str(ts),
        "X-TC-Version": version,
        "X-TC-Region": region,
    }


async def _tc_call(action: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """POST one Tencent Cloud API action; unwrap Response / raise on Error."""
    host = os.getenv("TENCENT_HUNYUAN_HOST", "hunyuan.intl.tencentcloudapi.com")
    service = os.getenv("TENCENT_HUNYUAN_SERVICE", "hunyuan")
    version = os.getenv("TENCENT_HUNYUAN_VERSION", "2023-09-01")
    region = os.getenv("TENCENT_HUNYUAN_REGION", "ap-singapore")
    secret_id = os.getenv("TENCENT_SECRET_ID", "")
    secret_key = os.getenv("TENCENT_SECRET_KEY", "")
    if not secret_id or not secret_key:
        raise HTTPException(
            status_code=500,
            detail="TENCENT_SECRET_ID / TENCENT_SECRET_KEY not configured on this host",
        )
    import httpx
    body = json.dumps(payload, separators=(",", ":")).encode()
    headers = _tc3_headers(action, body, host, service, version, region, secret_id, secret_key)
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(f"https://{host}/", content=body, headers=headers)
    try:
        data = r.json()
    except Exception:
        raise HTTPException(status_code=502, detail=f"Hunyuan API non-JSON reply (http {r.status_code}): {r.text[:200]}")
    resp = data.get("Response") or {}
    err = resp.get("Error")
    if err:
        logger.error(f"Hunyuan {action} error: {err}")
        raise HTTPException(
            status_code=502,
            detail={"error": "hunyuan_error", "code": err.get("Code"), "message": err.get("Message")},
        )
    return resp


# ── Endpoint models ─────────────────────────────────────────────────────

class Gen3DRequest(BaseModel):
    prompt: Optional[str] = Field(default=None, description="Text description (max 1024 chars). Exactly one of prompt / image_url / image_base64.")
    image_url: Optional[str] = Field(default=None, description="Image URL (JPG/PNG/WebP, 128-5000px, <8MB). Single object, plain background works best.")
    image_base64: Optional[str] = Field(default=None, description="Base64 image, same constraints as image_url")
    enable_pbr: bool = Field(default=False, description="Generate PBR materials")
    model_version: Optional[str] = Field(default=None, description="Hunyuan 3D model version: 3.0 (default) or 3.1")
    face_count: Optional[int] = Field(default=None, description="Polygon count 3000-1500000 (default 500000)")
    wait_for_result: bool = Field(default=True, description="true = poll until DONE/FAIL (sync); false = return job_id immediately")
    timeout_s: int = Field(default=480, description="Max seconds to wait when wait_for_result=true")
    save_to_storage: bool = Field(default=True, description="Persist result files to storage-api (Tencent URLs expire ~24h)")
    collection_id: Optional[str] = Field(default="ai-generated-3d", description="Storage collection for saved results")
    link_id: Optional[str] = Field(default=None)
    confirm_api_billing: Optional[bool] = Field(default=False, description="Required true — Tencent PAYG billing")


async def _save_files_to_storage(files: List[Dict[str, Any]], collection_id: Optional[str],
                                 link_id: Optional[str]) -> List[Dict[str, Any]]:
    """Download result files (size-capped) and persist them to storage-api."""
    import httpx
    storage_url = os.getenv("STORAGE_API_URL", "https://api-storage.arkturian.com")
    storage_key = os.getenv("STORAGE_API_KEY", "Inetpass1")
    max_bytes = int(os.getenv("GEN3D_MAX_FILE_MB", "200")) * 1024 * 1024
    from urllib.parse import urlparse
    saved = []
    # Flatten: each File3D has Url (the model, often a .zip containing the
    # OBJ/GLB) and optionally PreviewImageUrl — persist both.
    dl_items = []
    for f in files:
        ftype = (f.get("Type") or f.get("type") or "glb").lower()
        if f.get("Url") or f.get("url"):
            dl_items.append({"url": f.get("Url") or f.get("url"), "type": ftype})
        if f.get("PreviewImageUrl"):
            dl_items.append({"url": f["PreviewImageUrl"], "type": "preview"})
    for item in dl_items:
        url, ftype = item["url"], item["type"]
        # Extension from the URL path (Tencent serves OBJ results as .zip);
        # fall back to the declared type.
        ext = os.path.splitext(urlparse(url).path)[1].lstrip(".").lower() or ftype
        try:
            async with httpx.AsyncClient(timeout=300.0, follow_redirects=True) as client:
                async with client.stream("GET", url) as r:
                    r.raise_for_status()
                    clen = r.headers.get("content-length")
                    if clen and int(clen) > max_bytes:
                        logger.warning(f"gen3d: skip oversized result ({clen} bytes) {url[:80]}")
                        continue
                    buf, total = bytearray(), 0
                    async for chunk in r.aiter_bytes(65536):
                        total += len(chunk)
                        if total > max_bytes:
                            raise ValueError(f"result stream exceeded {max_bytes}-byte cap")
                        buf.extend(chunk)
                mime = {"glb": "model/gltf-binary", "gltf": "model/gltf+json",
                        "zip": "application/zip", "obj": "text/plain",
                        "fbx": "application/octet-stream",
                        "gif": "image/gif", "png": "image/png",
                        "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(ext, "application/octet-stream")
                data = {"is_public": "true", "analyze": "false", "ai_mode": "none",
                        "reuse_existing": "false"}
                if collection_id:
                    data["collection_id"] = collection_id
                if link_id:
                    data["link_id"] = link_id
                up = await client.post(
                    f"{storage_url}/storage/upload",
                    files={"file": (f"hunyuan3d_{uuid.uuid4().hex[:8]}.{ext}", bytes(buf), mime)},
                    data=data,
                    headers={"X-API-KEY": storage_key},
                )
                up.raise_for_status()
                sid = up.json().get("id")
                saved.append({
                    "type": ftype,
                    "storage_object_id": sid,
                    "storage_url": f"{storage_url}/storage/media/{sid}" if sid else None,
                    "tencent_url": url,
                })
                logger.info(f"gen3d: saved {ftype} ({total} bytes) -> media/{sid}")
        except Exception as e:
            logger.warning(f"gen3d: persist failed for {url[:80]}: {e}")
            saved.append({"type": ftype, "storage_object_id": None, "tencent_url": url,
                          "save_error": str(e)})
    return saved


@router.post("/gen3d")
async def generate_3d(request: Gen3DRequest, api_key: str = Depends(get_api_key)):
    """
    Generate a 3D model via Tencent Hunyuan (text-to-3D or image-to-3D).

    Provide exactly ONE of `prompt` / `image_url` / `image_base64`.
    PAYG — requires `confirm_api_billing: true` (default-deny, monthly cap).
    Sync by default (jobs typically take 1-5 min); set wait_for_result=false
    to get a job_id for polling via GET /ai/gen3d/status/{job_id}.
    Results are persisted to storage by default (Tencent URLs expire ~24h).
    """
    from .text_ai_routes import _check_api_billing_gate
    _check_api_billing_gate(request.confirm_api_billing, endpoint="hunyuan-3d")

    inputs = [x for x in (request.prompt, request.image_url, request.image_base64) if x]
    if len(inputs) != 1:
        raise HTTPException(
            status_code=422,
            detail="Provide exactly one of prompt / image_url / image_base64",
        )

    payload: Dict[str, Any] = {}
    if request.prompt:
        payload["Prompt"] = request.prompt
    if request.image_url:
        payload["ImageUrl"] = request.image_url
    if request.image_base64:
        payload["ImageBase64"] = request.image_base64
    if request.enable_pbr:
        payload["EnablePBR"] = True
    if request.model_version:
        payload["Model"] = request.model_version
    if request.face_count:
        payload["FaceCount"] = request.face_count

    resp = await _tc_call("SubmitHunyuanTo3DProJob", payload)
    job_id = resp.get("JobId")
    if not job_id:
        raise HTTPException(status_code=502, detail=f"Hunyuan submit returned no JobId: {resp}")
    logger.info(f"gen3d: submitted job {job_id} "
                f"({'text' if request.prompt else 'image'}-to-3d, pbr={request.enable_pbr})")

    if not request.wait_for_result:
        return {"job_id": job_id, "status": "WAIT"}

    deadline = time.monotonic() + max(30, request.timeout_s)
    status, files, q = "WAIT", [], {}
    while time.monotonic() < deadline:
        await asyncio.sleep(8)
        q = await _tc_call("QueryHunyuanTo3DProJob", {"JobId": job_id})
        status = q.get("Status") or "WAIT"
        if status in ("DONE", "FAIL"):
            break
    if status == "FAIL":
        raise HTTPException(
            status_code=502,
            detail={"error": "hunyuan_job_failed", "job_id": job_id,
                    "code": q.get("ErrorCode"), "message": q.get("ErrorMessage")},
        )
    if status != "DONE":
        return {"job_id": job_id, "status": status,
                "hint": f"still {status} after {request.timeout_s}s — poll GET /ai/gen3d/status/{job_id}"}

    files = q.get("ResultFile3Ds") or []
    result: Dict[str, Any] = {"job_id": job_id, "status": "DONE", "files": files}
    if request.save_to_storage and files:
        result["saved"] = await _save_files_to_storage(files, request.collection_id, request.link_id)
    return result


@router.get("/gen3d/status/{job_id}")
async def gen3d_status(job_id: str, save_to_storage: bool = False,
                       collection_id: Optional[str] = "ai-generated-3d",
                       link_id: Optional[str] = None,
                       api_key: str = Depends(get_api_key)):
    """Poll a Hunyuan 3D job. Optionally persist results on DONE (?save_to_storage=true)."""
    q = await _tc_call("QueryHunyuanTo3DProJob", {"JobId": job_id})
    status = q.get("Status") or "WAIT"
    out: Dict[str, Any] = {"job_id": job_id, "status": status,
                           "files": q.get("ResultFile3Ds") or []}
    if status == "FAIL":
        out["error_code"] = q.get("ErrorCode")
        out["error_message"] = q.get("ErrorMessage")
    if status == "DONE" and save_to_storage and out["files"]:
        out["saved"] = await _save_files_to_storage(out["files"], collection_id, link_id)
    return out
