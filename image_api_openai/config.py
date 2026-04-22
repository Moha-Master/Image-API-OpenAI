import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Config:
    def __init__(self) -> None:
        self._load_config()

    def _load_config(self) -> None:
        config_path = Path("config.yaml")
        logger.info("Loading configuration from %s", config_path)

        if not config_path.exists():
            raise FileNotFoundError(
                "Configuration file config.yaml not found. "
                "Please create it from config.yaml.example"
            )

        with config_path.open("r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}

        self.API_KEYS: Set[str] = set(config_data.get("api_keys", []))

        providers = config_data.get("providers", {})
        modelscope = providers.get("modelscope", {})
        siliconflow = providers.get("siliconflow", {})

        self.MODELSCOPE_ENABLED: bool = bool(modelscope.get("enabled", True))
        self.MODELSCOPE_BASE_URL: str = modelscope.get(
            "base_url", "https://api-inference.modelscope.cn/v1"
        )
        self.MODELSCOPE_API_KEY: str = modelscope.get("api_key", "")
        self.MODELSCOPE_ASYNC_MODE: bool = bool(modelscope.get("async_mode", False))
        self.MODELSCOPE_TASK_TYPE: str = modelscope.get("task_type", "image_generation")

        self.SILICONFLOW_ENABLED: bool = bool(siliconflow.get("enabled", True))
        self.SILICONFLOW_BASE_URL: str = siliconflow.get(
            "base_url", "https://api.siliconflow.cn/v1"
        )
        self.SILICONFLOW_API_KEY: str = siliconflow.get("api_key", "")

        defaults = config_data.get("defaults", {})
        self.DEFAULT_SIZE: str = defaults.get("size", "1024x1024")
        self.DEFAULT_N: int = int(defaults.get("n", 1))
        self.DEFAULT_RESPONSE_FORMAT: str = defaults.get("response_format", "b64_json")

        supported = config_data.get("supported", {})
        self.SUPPORTED_SIZES: List[str] = supported.get(
            "sizes",
            [
                "1024x1024",
                "960x1280",
                "768x1024",
                "720x1280",
                "1328x1328",
                "1664x928",
                "928x1664",
            ],
        )
        self.SUPPORTED_RESPONSE_FORMATS: Set[str] = set(
            supported.get("response_formats", ["url", "b64_json"])
        )
        self.STATIC_MODELS: Dict[str, List[str]] = supported.get(
            "static_models",
            {"modelscope": [], "siliconflow": []},
        )

        self.PROVIDER_MODEL_ALIASES: Dict[str, Dict[str, Dict[str, Any]]] = {
            "modelscope": self._parse_provider_aliases(modelscope),
            "siliconflow": self._parse_provider_aliases(siliconflow),
        }
        self.ALIAS_INDEX: Dict[str, List[Dict[str, Any]]] = self._build_alias_index()

        self.REQUEST_TIMEOUT_SECONDS: int = int(
            config_data.get("network", {}).get("timeout_seconds", 120)
        )

    def _parse_provider_aliases(self, provider_cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        aliases_cfg = provider_cfg.get("aliases", {})
        aliases: Dict[str, Dict[str, Any]] = {}
        if not isinstance(aliases_cfg, dict):
            return aliases

        for model_id, alias_value in aliases_cfg.items():
            if not isinstance(model_id, str) or not model_id:
                continue
            if isinstance(alias_value, str):
                alias_name = alias_value.strip()
                priority = 0
            elif isinstance(alias_value, dict):
                alias_name = str(alias_value.get("alias") or alias_value.get("name") or "").strip()
                try:
                    priority = int(alias_value.get("priority", 0))
                except Exception:
                    priority = 0
            else:
                continue

            if not alias_name:
                continue

            aliases[model_id] = {
                "alias": alias_name,
                "priority": priority,
            }
        return aliases

    def _build_alias_index(self) -> Dict[str, List[Dict[str, Any]]]:
        index: Dict[str, List[Dict[str, Any]]] = {}
        for provider, model_map in self.PROVIDER_MODEL_ALIASES.items():
            for model_id, alias_info in model_map.items():
                alias = alias_info["alias"]
                index.setdefault(alias, []).append(
                    {
                        "provider": provider,
                        "model": model_id,
                        "priority": int(alias_info.get("priority", 0)),
                    }
                )
        return index

    def parse_prefixed_model(self, model: str) -> Optional[Tuple[str, str]]:
        for provider in ("modelscope", "siliconflow"):
            prefix = f"{provider}/"
            if model.startswith(prefix):
                provider_model = model[len(prefix) :].strip()
                if provider_model:
                    return provider, provider_model
        return None

    def resolve_model(self, requested_model: str, user_provider: Optional[str]) -> Tuple[str, str]:
        prefixed = self.parse_prefixed_model(requested_model)
        normalized_user_provider: Optional[str] = None
        if user_provider:
            normalized_user_provider = user_provider.strip().lower()
            if normalized_user_provider not in {"modelscope", "siliconflow"}:
                raise ValueError("provider must be one of: modelscope, siliconflow")

        if prefixed:
            prefixed_provider, provider_model = prefixed
            if normalized_user_provider and normalized_user_provider != prefixed_provider:
                raise ValueError("provider conflicts with model prefix")
            return prefixed_provider, provider_model

        alias_candidates = self.ALIAS_INDEX.get(requested_model, [])
        if alias_candidates:
            candidates = alias_candidates
            if normalized_user_provider:
                candidates = [c for c in alias_candidates if c["provider"] == normalized_user_provider]
            if not candidates:
                raise ValueError("alias exists but not for requested provider")

            max_priority = max(int(c["priority"]) for c in candidates)
            top = [c for c in candidates if int(c["priority"]) == max_priority]
            selected = random.choice(top)
            return selected["provider"], selected["model"]

        if normalized_user_provider:
            return normalized_user_provider, requested_model

        raise ValueError(
            "Could not resolve provider for model. "
            "Use model prefix (e.g. 'modelscope/Qwen/Qwen-Image') or provide 'provider' parameter, "
            "or configure an alias for this model."
        )
