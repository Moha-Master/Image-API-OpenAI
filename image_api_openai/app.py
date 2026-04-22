import base64
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Set

import requests
from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel

from .config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config: Optional[Config] = None


def initialize_config() -> None:
    global config
    config = Config()
    logger.info(
        "Configuration initialized: modelscope_enabled=%s, siliconflow_enabled=%s",
        config.MODELSCOPE_ENABLED,
        config.SILICONFLOW_ENABLED,
    )


class ImageGenerationRequest(BaseModel):
    model: Optional[str] = None
    prompt: str
    negative_prompt: Optional[str] = None
    size: Optional[str] = None
    n: Optional[int] = None
    response_format: Optional[str] = None
    provider: Optional[str] = None
    user: Optional[str] = None

    class Config:
        extra = "allow"


app = FastAPI(
    title="Image API OpenAI Proxy",
    description="Convert ModelScope and SiliconFlow image APIs into OpenAI-compatible format",
    version="0.1.0",
)


@app.on_event("startup")
async def startup_event() -> None:
    initialize_config()


def _check_auth(authorization: Optional[str]) -> None:
    if config is None:
        raise HTTPException(status_code=500, detail="Configuration not initialized")
    if not config.API_KEYS:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    api_key = authorization.split(" ", 1)[1]
    if api_key not in config.API_KEYS:
        raise HTTPException(status_code=403, detail="Forbidden")


def _as_openai_response(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "created": int(time.time()),
        "data": items,
    }


def _download_image_as_b64(image_url: str) -> str:
    if config is None:
        raise RuntimeError("Configuration not initialized")

    headers = {
        "User-Agent": "image-api-openai/0.1",
        "Accept": "image/*,*/*;q=0.8",
    }
    resp = requests.get(
        image_url,
        headers=headers,
        timeout=config.REQUEST_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    content_type = (resp.headers.get("content-type") or "").lower()
    if not content_type.startswith("image/"):
        logger.warning(
            "downloaded url content-type is not image: content-type=%s url=%s",
            content_type,
            image_url,
        )
    return base64.b64encode(resp.content).decode("ascii")


def _safe_preview(data: Any, max_len: int = 500) -> str:
    text = str(data)
    if len(text) <= max_len:
        return text
    return text[:max_len] + "...(truncated)"


def _request_modelscope(payload: Dict[str, Any]) -> Dict[str, Any]:
    if config is None:
        raise RuntimeError("Configuration not initialized")
    if not config.MODELSCOPE_ENABLED:
        raise HTTPException(status_code=400, detail="ModelScope provider is disabled")
    if not config.MODELSCOPE_API_KEY:
        raise HTTPException(status_code=500, detail="ModelScope api_key is not configured")

    url = f"{config.MODELSCOPE_BASE_URL.rstrip('/')}/images/generations"
    headers = {
        "Authorization": f"Bearer {config.MODELSCOPE_API_KEY}",
        "Content-Type": "application/json",
    }
    if config.MODELSCOPE_ASYNC_MODE:
        headers["X-ModelScope-Async-Mode"] = "true"
        headers["X-ModelScope-Task-Type"] = config.MODELSCOPE_TASK_TYPE

    resp = requests.post(
        url,
        json=payload,
        headers=headers,
        timeout=config.REQUEST_TIMEOUT_SECONDS,
    )

    logger.info(
        "ModelScope response status=%s headers.x-request-id=%s",
        resp.status_code,
        resp.headers.get("x-request-id") or resp.headers.get("x-modelscope-request-id"),
    )

    if resp.status_code >= 400:
        detail = resp.text
        try:
            detail = resp.json()
        except Exception:
            pass
        logger.error("ModelScope request failed: %s", _safe_preview(detail))
        raise HTTPException(status_code=resp.status_code, detail=detail)

    data = resp.json()
    if "images" in data:
        return data

    task_id = data.get("task_id")
    if task_id:
        logger.info("ModelScope async task created: task_id=%s", task_id)
        task_url = f"{config.MODELSCOPE_BASE_URL.rstrip('/')}/tasks/{task_id}"
        poll_headers = {
            "Authorization": f"Bearer {config.MODELSCOPE_API_KEY}",
            "Content-Type": "application/json",
            "X-ModelScope-Task-Type": config.MODELSCOPE_TASK_TYPE,
        }
        deadline = time.time() + config.REQUEST_TIMEOUT_SECONDS
        polling_interval = 5  # 初始轮询间隔5秒
        start_time = time.time()
        while time.time() < deadline:
            task_resp = requests.get(
                task_url,
                headers=poll_headers,
                timeout=30,
            )
            if task_resp.status_code >= 400:
                detail = task_resp.text
                try:
                    detail = task_resp.json()
                except Exception:
                    pass
                logger.error("ModelScope task polling failed: %s", _safe_preview(detail))
                raise HTTPException(status_code=task_resp.status_code, detail=detail)

            task_data = task_resp.json()
            status = str(task_data.get("task_status", "")).upper()
            logger.info("ModelScope task polling: task_id=%s status=%s", task_id, status)
            if status in {"SUCCEED", "SUCCESS"}:
                output_images = task_data.get("output_images", [])
                logger.info("ModelScope task succeeded: task_id=%s output_count=%s", task_id, len(output_images))
                return {
                    "images": [{"url": u} for u in output_images if isinstance(u, str)]
                }
            if status in {"FAILED", "CANCELED", "CANCELLED"}:
                logger.error("ModelScope task failed: task_id=%s detail=%s", task_id, _safe_preview(task_data))
                raise HTTPException(status_code=502, detail=task_data)
            
            # 超过1分钟后增加轮询间隔到15秒
            if time.time() - start_time > 60:
                polling_interval = 15
            
            time.sleep(polling_interval)

        raise HTTPException(status_code=504, detail="ModelScope task polling timed out")

    return data


def _request_siliconflow(payload: Dict[str, Any]) -> Dict[str, Any]:
    if config is None:
        raise RuntimeError("Configuration not initialized")
    if not config.SILICONFLOW_ENABLED:
        raise HTTPException(status_code=400, detail="SiliconFlow provider is disabled")
    if not config.SILICONFLOW_API_KEY:
        raise HTTPException(status_code=500, detail="SiliconFlow api_key is not configured")

    url = f"{config.SILICONFLOW_BASE_URL.rstrip('/')}/images/generations"
    headers = {
        "Authorization": f"Bearer {config.SILICONFLOW_API_KEY}",
        "Content-Type": "application/json",
    }

    resp = requests.post(
        url,
        json=payload,
        headers=headers,
        timeout=config.REQUEST_TIMEOUT_SECONDS,
    )

    logger.info(
        "SiliconFlow response status=%s trace_id=%s",
        resp.status_code,
        resp.headers.get("x-siliconcloud-trace-id"),
    )

    if resp.status_code >= 400:
        detail = resp.text
        try:
            detail = resp.json()
        except Exception:
            pass
        logger.error("SiliconFlow request failed: %s", _safe_preview(detail))
        raise HTTPException(status_code=resp.status_code, detail=detail)

    return resp.json()


def _list_models_siliconflow() -> List[str]:
    if config is None:
        raise RuntimeError("Configuration not initialized")
    if not config.SILICONFLOW_ENABLED or not config.SILICONFLOW_API_KEY:
        return []

    url = f"{config.SILICONFLOW_BASE_URL.rstrip('/')}/models"
    headers = {
        "Authorization": f"Bearer {config.SILICONFLOW_API_KEY}",
        "Content-Type": "application/json",
    }
    params = {"type": "text-to-image"}

    try:
        resp = requests.get(
            url,
            headers=headers,
            params=params,
            timeout=config.REQUEST_TIMEOUT_SECONDS,
        )
        if resp.status_code >= 400:
            logger.warning("SiliconFlow /models failed with status %s", resp.status_code)
            return []
        data = resp.json()
        model_ids: List[str] = []
        for item in data.get("data", []):
            if isinstance(item, dict) and isinstance(item.get("id"), str):
                model_ids.append(item["id"])
        return model_ids
    except Exception as e:
        logger.warning("SiliconFlow /models request failed: %s", e)
        return []


def _list_models_modelscope() -> List[str]:
    if config is None:
        raise RuntimeError("Configuration not initialized")
    static_models = config.STATIC_MODELS.get("modelscope", [])
    return [m for m in static_models if isinstance(m, str)]


def _collect_provider_models() -> Dict[str, List[str]]:
    if config is None:
        raise RuntimeError("Configuration not initialized")

    models: Dict[str, Set[str]] = {"modelscope": set(), "siliconflow": set()}

    for provider in ("modelscope", "siliconflow"):
        for model_id in config.STATIC_MODELS.get(provider, []):
            if isinstance(model_id, str) and model_id:
                models[provider].add(model_id)

    for provider, model_map in config.PROVIDER_MODEL_ALIASES.items():
        for model_id in model_map.keys():
            if model_id:
                models[provider].add(model_id)

    for model_id in _list_models_siliconflow():
        if model_id:
            models["siliconflow"].add(model_id)

    return {
        "modelscope": sorted(models["modelscope"]),
        "siliconflow": sorted(models["siliconflow"]),
    }


def _build_model_item(model_id: str, owned_by: str) -> Dict[str, Any]:
    return {
        "id": model_id,
        "object": "model",
        "created": 0,
        "owned_by": owned_by,
    }


def _get_form_str(form, key: str) -> Optional[str]:
    val = form.get(key)
    return val if isinstance(val, str) else None


def _extract_images(
    data: Dict[str, Any],
    response_format: str,
    request_id: str,
) -> List[Dict[str, Any]]:
    images: List[Dict[str, Any]] = []
    for item in data.get("images", []):
        url = item.get("url") if isinstance(item, dict) else None
        b64 = item.get("b64_json") if isinstance(item, dict) else None
        if response_format == "b64_json" and not b64 and url:
            try:
                b64 = _download_image_as_b64(url)
            except Exception as e:
                raise HTTPException(status_code=502, detail=f"Download image failed: {e}")
        if response_format == "b64_json":
            if b64:
                preview = f"{b64[:80]}...{b64[-20:]}" if len(b64) > 100 else b64
                logger.info("[%s] extracted b64_json preview: %s (total %d chars)", request_id, preview, len(b64))
                images.append({"b64_json": b64})
        else:
            if url:
                logger.info("[%s] image url=%s", request_id, url)
                images.append({"url": url})
    return images


@app.post("/v1/images/generations")
async def create_image(
    request: ImageGenerationRequest,
    authorization: Optional[str] = Header(None),
) -> Dict[str, Any]:
    request_id = str(uuid.uuid4())[:8]
    logger.info("[%s] /v1/images/generations request received", request_id)
    _check_auth(authorization)

    if config is None:
        raise HTTPException(status_code=500, detail="Configuration not initialized")

    n = request.n if request.n is not None else config.DEFAULT_N
    size = request.size or config.DEFAULT_SIZE
    response_format = request.response_format or config.DEFAULT_RESPONSE_FORMAT
    requested_model = request.model

    if not requested_model:
        raise HTTPException(status_code=400, detail="model is required")

    if n < 1:
        raise HTTPException(status_code=400, detail="n must be >= 1")
    if response_format not in config.SUPPORTED_RESPONSE_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported response_format: {response_format}",
        )

    try:
        provider, model = config.resolve_model(
            requested_model=requested_model,
            user_provider=request.provider,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    logger.info(
        "[%s] resolved provider=%s requested_model=%s provider_model=%s size=%s n=%s response_format=%s",
        request_id,
        provider,
        requested_model,
        model,
        size,
        n,
        response_format,
    )

    payload: Dict[str, Any] = request.dict(exclude_none=True)
    payload.pop("provider", None)
    payload["model"] = model
    payload["prompt"] = request.prompt
    logger.info(
        "[%s] upstream payload keys=%s",
        request_id,
        sorted(payload.keys()),
    )

    images: List[Dict[str, Any]] = []

    if provider == "modelscope":
        payload["n"] = n
        payload["size"] = size
        data = _request_modelscope(payload)
        logger.info("[%s] modelscope response keys=%s", request_id, sorted(data.keys()))
        images = _extract_images(data, response_format, request_id)

    elif provider == "siliconflow":
        payload["image_size"] = size
        payload["batch_size"] = n
        payload.pop("size", None)
        payload.pop("n", None)
        data = _request_siliconflow(payload)
        logger.info("[%s] siliconflow response keys=%s", request_id, sorted(data.keys()))
        images = _extract_images(data, response_format, request_id)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported provider: {provider}")

    if not images:
        logger.error("[%s] no images extracted from provider response", request_id)
        raise HTTPException(status_code=502, detail="No images found in provider response")

    logger.info("[%s] success image_count=%s", request_id, len(images))
    return _as_openai_response(images)


@app.post("/v1/images/edits")
async def create_image_edit(
    request: Request,
    authorization: Optional[str] = Header(None),
) -> Dict[str, Any]:
    request_id = str(uuid.uuid4())[:8]
    logger.info("[%s] /v1/images/edits request received", request_id)
    _check_auth(authorization)

    if config is None:
        raise HTTPException(status_code=500, detail="Configuration not initialized")

    content_type = request.headers.get("content-type", "").lower()
    logger.info("[%s] edit content-type=%s", request_id, content_type)

    if "multipart/form-data" in content_type:
        form = await request.form()
        form_keys = list(form.keys())
        logger.info("[%s] edit multipart form keys=%s", request_id, form_keys)

        requested_model = _get_form_str(form, "model")
        logger.info("[%s] edit multipart model=%s", request_id, requested_model)

        prompt = _get_form_str(form, "prompt")
        if not prompt:
            logger.warning("[%s] edit validation failed: prompt is missing", request_id)
            raise HTTPException(status_code=400, detail="prompt is required")
        provider = _get_form_str(form, "provider")
        size = _get_form_str(form, "size") or config.DEFAULT_SIZE
        n_str = _get_form_str(form, "n")
        n = int(n_str) if n_str else config.DEFAULT_N
        response_format = _get_form_str(form, "response_format") or config.DEFAULT_RESPONSE_FORMAT

        # 兼容 OpenAI 标准的 image[] 和常规的 image 字段名
        image_file = form.get("image") or form.get("image[]")
        logger.info("[%s] edit multipart image field type=%s has_read=%s", request_id, type(image_file).__name__, hasattr(image_file, "read"))
        if image_file is None or not hasattr(image_file, "read"):
            logger.warning("[%s] edit validation failed: image is not a file upload", request_id)
            raise HTTPException(status_code=400, detail="image is required (file upload)")
        image_bytes = await image_file.read()
        if not image_bytes:
            logger.warning("[%s] edit validation failed: image file is empty", request_id)
            raise HTTPException(status_code=400, detail="image file is empty")
        img_ct = getattr(image_file, "content_type", None) or "image/png"
        image_value = f"data:{img_ct};base64,{base64.b64encode(image_bytes).decode('ascii')}"
        logger.info("[%s] edit multipart image processed: content_type=%s size=%d bytes", request_id, img_ct, len(image_bytes))

        mask_value = None
        mask_file = form.get("mask") or form.get("mask[]")
        if mask_file is not None and hasattr(mask_file, "read"):
            mask_bytes = await mask_file.read()
            if mask_bytes:
                mask_ct = getattr(mask_file, "content_type", None) or "image/png"
                mask_value = f"data:{mask_ct};base64,{base64.b64encode(mask_bytes).decode('ascii')}"
                logger.info("[%s] edit multipart mask processed: content_type=%s size=%d bytes", request_id, mask_ct, len(mask_bytes))

        extra_payload: Dict[str, Any] = {}
    else:
        body = await request.json()
        body_preview = {k: v if k not in ("image", "mask") else f"<{type(v).__name__}:{len(str(v))} chars>" for k, v in body.items()}
        logger.info("[%s] edit json body keys=%s preview=%s", request_id, sorted(body.keys()), body_preview)

        requested_model = body.get("model")
        prompt = body.get("prompt")
        if not prompt:
            logger.warning("[%s] edit validation failed: prompt is missing", request_id)
            raise HTTPException(status_code=400, detail="prompt is required")
        provider = body.get("provider")
        size = body.get("size") or config.DEFAULT_SIZE
        n = int(body.get("n", config.DEFAULT_N))
        response_format = body.get("response_format") or config.DEFAULT_RESPONSE_FORMAT
        image_value = body.get("image")
        if not image_value or not isinstance(image_value, str):
            logger.warning("[%s] edit validation failed: image is missing or not a string, type=%s", request_id, type(image_value).__name__)
            raise HTTPException(status_code=400, detail="image is required (base64 or URL)")
        mask_value = body.get("mask")

        extra_payload = {
            k: v for k, v in body.items()
            if k not in {"model", "prompt", "image", "mask", "size", "n", "response_format", "provider", "user"}
        }

    if not requested_model:
        logger.warning("[%s] edit validation failed: model is missing", request_id)
        raise HTTPException(status_code=400, detail="model is required")
    if n < 1:
        logger.warning("[%s] edit validation failed: n=%s is invalid", request_id, n)
        raise HTTPException(status_code=400, detail="n must be >= 1")
    if response_format not in config.SUPPORTED_RESPONSE_FORMATS:
        logger.warning("[%s] edit validation failed: unsupported response_format=%s", request_id, response_format)
        raise HTTPException(status_code=400, detail=f"Unsupported response_format: {response_format}")

    logger.info(
        "[%s] edit resolving model: requested_model=%s provider=%s",
        request_id, requested_model, provider,
    )
    try:
        resolved_provider, resolved_model = config.resolve_model(
            requested_model=requested_model,
            user_provider=provider,
        )
    except ValueError as e:
        logger.warning("[%s] edit model resolution failed: %s", request_id, e)
        raise HTTPException(status_code=400, detail=str(e))

    logger.info(
        "[%s] edit resolved provider=%s requested_model=%s provider_model=%s size=%s n=%s response_format=%s",
        request_id, resolved_provider, requested_model, resolved_model, size, n, response_format,
    )

    payload: Dict[str, Any] = {
        "model": resolved_model,
        "prompt": prompt,
        "image": image_value,
        **extra_payload,
    }
    if mask_value:
        payload["mask"] = mask_value

    logger.info("[%s] edit upstream payload keys=%s", request_id, sorted(payload.keys()))

    images: List[Dict[str, Any]] = []

    if resolved_provider == "modelscope":
        payload["n"] = n
        payload["size"] = size
        # ModelScope AIGC API 使用 image_url 而非 image
        if "image" in payload:
            payload["image_url"] = payload.pop("image")
        if "mask" in payload:
            payload.pop("mask", None)  # ModelScope 不支持 mask
        data = _request_modelscope(payload)
        logger.info("[%s] modelscope edit response keys=%s", request_id, sorted(data.keys()))
        images = _extract_images(data, response_format, request_id)

    elif resolved_provider == "siliconflow":
        payload["image_size"] = size
        payload["batch_size"] = n
        payload.pop("size", None)
        payload.pop("n", None)
        # SiliconFlow 支持 image, image2, image3，mask 被忽略
        if "mask" in payload:
            payload.pop("mask", None)
        data = _request_siliconflow(payload)
        logger.info("[%s] siliconflow edit response keys=%s", request_id, sorted(data.keys()))
        images = _extract_images(data, response_format, request_id)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported provider: {resolved_provider}")

    if not images:
        logger.error("[%s] no images extracted from edit provider response", request_id)
        raise HTTPException(status_code=502, detail="No images found in provider response")

    logger.info("[%s] edit success image_count=%s", request_id, len(images))
    return _as_openai_response(images)


@app.get("/v1/providers")
async def list_providers() -> Dict[str, Any]:
    if config is None:
        raise HTTPException(status_code=500, detail="Configuration not initialized")
    return {
        "providers": [
            {
                "name": "modelscope",
                "enabled": config.MODELSCOPE_ENABLED,
                "base_url": config.MODELSCOPE_BASE_URL,
            },
            {
                "name": "siliconflow",
                "enabled": config.SILICONFLOW_ENABLED,
                "base_url": config.SILICONFLOW_BASE_URL,
            },
        ]
    }


@app.get("/v1/models")
async def list_models(
    authorization: Optional[str] = Header(None),
) -> Dict[str, Any]:
    _check_auth(authorization)

    if config is None:
        raise HTTPException(status_code=500, detail="Configuration not initialized")

    models: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    provider_models = _collect_provider_models()
    for provider, model_ids in provider_models.items():
        for model_id in model_ids:
            exposed_model_id = f"{provider}/{model_id}"
            if exposed_model_id in seen:
                continue
            seen.add(exposed_model_id)
            models.append(_build_model_item(exposed_model_id, provider))

    for alias_name in sorted(config.ALIAS_INDEX.keys()):
        if alias_name in seen:
            continue
        seen.add(alias_name)
        models.append(_build_model_item(alias_name, "alias"))

    return {
        "object": "list",
        "data": models,
    }
