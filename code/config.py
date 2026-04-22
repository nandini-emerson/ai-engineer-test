
# python
import os
import logging
import json
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# Load .env file FIRST (must happen before any os.getenv() calls)
load_dotenv()

logger = logging.getLogger(__name__)

class Config:
    """Centralized configuration for the agent.

    - All values are populated from Azure Key Vault (when enabled) or from
      environment variables (.env). If a variable is not found in either
      source it is set to an empty string "" (or None for explicitly optional
      values) and a WARNING is emitted referencing the .env file per policy.
    """

    # Key Vault cache
    _kv_secrets: Dict[str, str] = {}

    # Key Vault secret map — copy of relevant entries from platform reference.
    # Maps (Config attribute name) -> ("key-vault-secret-name.json_key" | "secret-name")
    KEY_VAULT_SECRET_MAP = [
        # LLM API Keys
        ("AZURE_OPENAI_API_KEY", "openai-secrets.gpt-4.1"),
        ("OPENAI_API_KEY", "aba-openai-secret.openai_api_key"),

        # Azure OpenAI endpoint (embedding / deployments)
        ("AZURE_OPENAI_ENDPOINT", "kb-secrets.azure_openai_endpoint"),

        # Azure Content Safety
        ("AZURE_CONTENT_SAFETY_ENDPOINT", "azure-content-safety-secrets.azure_content_safety_endpoint"),
        ("AZURE_CONTENT_SAFETY_KEY", "azure-content-safety-secrets.azure_content_safety_key"),

        # Observability Azure SQL Database secrets
        ("OBS_AZURE_SQL_SERVER", "agentops-secrets.obs_sql_endpoint"),
        ("OBS_AZURE_SQL_DATABASE", "agentops-secrets.obs_azure_sql_database"),
        ("OBS_AZURE_SQL_PORT", "agentops-secrets.obs_port"),
        ("OBS_AZURE_SQL_USERNAME", "agentops-secrets.obs_sql_username"),
        ("OBS_AZURE_SQL_PASSWORD", "agentops-secrets.obs_sql_password"),
        ("OBS_AZURE_SQL_SCHEMA", "agentops-secrets.obs_azure_sql_schema"),
    ]

    # Models that do not accept temperature / max_tokens will be checked at runtime
    _MAX_TOKENS_UNSUPPORTED = {
        "gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-5.1-chat",
        "o1", "o1-mini", "o1-preview", "o3", "o3-mini", "o3-pro", "o4-mini"
    }

    _TEMPERATURE_UNSUPPORTED = set(_MAX_TOKENS_UNSUPPORTED)

    @classmethod
    def _load_keyvault_secrets(cls) -> Dict[str, str]:
        """
        Load secrets from Azure Key Vault into cls._kv_secrets.

        Only runs when USE_KEY_VAULT is True and KEY_VAULT_URI is set (non-empty).
        """
        # If Key Vault is not configured properly, return empty dict
        try:
            use_kv = bool(getattr(cls, "USE_KEY_VAULT", False))
            kv_uri = getattr(cls, "KEY_VAULT_URI", "") or ""
        except Exception:
            return {}

        if not use_kv or not kv_uri:
            logger.debug("Key Vault not enabled or KEY_VAULT_URI empty; skipping Key Vault load")
            return {}

        # Decide credential type
        AZURE_USE_DEFAULT_CREDENTIAL = os.getenv("AZURE_USE_DEFAULT_CREDENTIAL", "").lower() in ("true", "1", "yes")
        credential = None
        try:
            if AZURE_USE_DEFAULT_CREDENTIAL:
                from azure.identity import DefaultAzureCredential
                credential = DefaultAzureCredential()
            else:
                from azure.identity import ClientSecretCredential
                tenant_id = os.getenv("AZURE_TENANT_ID", "")
                client_id = os.getenv("AZURE_CLIENT_ID", "")
                client_secret = os.getenv("AZURE_CLIENT_SECRET", "")
                if not (tenant_id and client_id and client_secret):
                    logging.warning("Service Principal credentials incomplete. Key Vault access will fail.")
                    return {}
                credential = ClientSecretCredential(
                    tenant_id=tenant_id,
                    client_id=client_id,
                    client_secret=client_secret
                )
        except Exception as e:
            logger.warning("Failed to create Azure credential for Key Vault access: %s", e)
            return {}

        try:
            from azure.keyvault.secrets import SecretClient
            client = SecretClient(vault_url=kv_uri, credential=credential)
        except Exception as e:
            logger.warning("Failed to construct Key Vault SecretClient: %s", e)
            return {}

        # Group KEY_VAULT_SECRET_MAP by secret name to minimize round-trips
        by_secret: Dict[str, List[tuple]] = {}
        for attr_name, secret_ref in getattr(cls, "KEY_VAULT_SECRET_MAP", []) or []:
            if "." in secret_ref:
                secret_name, json_key = secret_ref.split(".", 1)
            else:
                secret_name, json_key = secret_ref, None
            by_secret.setdefault(secret_name, []).append((attr_name, json_key))

        for secret_name, refs in by_secret.items():
            try:
                secret_bundle = client.get_secret(secret_name)
                if not secret_bundle or not getattr(secret_bundle, "value", None):
                    logger.debug("Key Vault: secret '%s' empty or missing", secret_name)
                    continue
                raw_value = secret_bundle.value
                # Strip possible BOM markers
                for _bom in ("\ufeff", "\xef\xbb\xbf"):
                    if raw_value.startswith(_bom):
                        raw_value = raw_value[len(_bom):]
                        break

                has_json_key = any(json_key is not None for _, json_key in refs)
                if has_json_key:
                    try:
                        parsed = json.loads(raw_value)
                        if not isinstance(parsed, dict):
                            logger.debug("Key Vault: secret '%s' parsed value is not a JSON object", secret_name)
                            continue
                        for attr_name, json_key in refs:
                            if json_key is None:
                                continue
                            try:
                                val = parsed.get(json_key)
                                if val is not None and val != "":
                                    cls._kv_secrets[attr_name] = str(val)
                                    logger.debug("Key Vault: loaded %s from %s.%s", attr_name, secret_name, json_key)
                                else:
                                    logger.debug("Key Vault: json key '%s' not found in secret '%s'", json_key, secret_name)
                            except Exception:
                                logger.debug("Key Vault: failed to extract json key '%s' from secret '%s'", json_key, secret_name)
                                continue
                    except json.JSONDecodeError:
                        logger.debug("Key Vault: secret '%s' value is not valid JSON", secret_name)
                        continue
                else:
                    # Single-value secret: assign to all refs that point to this secret (first wins)
                    for attr_name, json_key in refs:
                        if json_key is None and raw_value:
                            cls._kv_secrets[attr_name] = str(raw_value)
                            logger.debug("Key Vault: loaded %s from %s", attr_name, secret_name)
            except Exception as e:
                # Per-secret failures should not crash startup
                logger.debug("Key Vault: failed to fetch secret '%s': %s", secret_name, e)
                continue

        return cls._kv_secrets

    @classmethod
    def _validate_api_keys(cls) -> None:
        """
        Ensure provider-specific API key is present. Raises ValueError for missing key.
        """
        provider = (getattr(cls, "MODEL_PROVIDER", "") or "").lower()
        if provider == "azure":
            key = getattr(cls, "AZURE_OPENAI_API_KEY", "") or ""
            if not key:
                raise ValueError("AZURE_OPENAI_API_KEY is required for provider 'azure'")
        elif provider == "openai":
            key = getattr(cls, "OPENAI_API_KEY", "") or ""
            if not key:
                raise ValueError("OPENAI_API_KEY is required for provider 'openai'")
        elif provider == "anthropic":
            key = getattr(cls, "ANTHROPIC_API_KEY", "") or ""
            if not key:
                raise ValueError("ANTHROPIC_API_KEY is required for provider 'anthropic'")
        elif provider == "google":
            key = getattr(cls, "GOOGLE_API_KEY", "") or ""
            if not key:
                raise ValueError("GOOGLE_API_KEY is required for provider 'google'")
        # If provider is unspecified or unknown, do not raise here; let runtime decide.

    @classmethod
    def get_llm_kwargs(cls) -> Dict[str, Any]:
        """
        Return LLM kwargs suitable for chat.completions.create() style calls.
        Dynamically switches between temperature vs omit and max_tokens vs max_completion_tokens
        depending on the current model.
        """
        kwargs: Dict[str, Any] = {}
        model_lower = (getattr(cls, "LLM_MODEL", "") or "").lower()
        temp_val = getattr(cls, "LLM_TEMPERATURE", None)
        max_tokens_val = getattr(cls, "LLM_MAX_TOKENS", None)

        if temp_val is not None:
            # Only include temperature when model supports it
            if not any(model_lower.startswith(m) for m in cls._TEMPERATURE_UNSUPPORTED):
                kwargs["temperature"] = temp_val

        if max_tokens_val is not None:
            if any(model_lower.startswith(m) for m in cls._MAX_TOKENS_UNSUPPORTED):
                kwargs["max_completion_tokens"] = max_tokens_val
            else:
                kwargs["max_tokens"] = max_tokens_val

        return kwargs

    @classmethod
    def validate(cls) -> None:
        """Public validation entry point. Raises on missing required API keys."""
        cls._validate_api_keys()


def _initialize_config() -> None:
    """
    Initialize Config class attributes.

    Flow:
      1. Read USE_KEY_VAULT, KEY_VAULT_URI, AZURE_USE_DEFAULT_CREDENTIAL from .env
      2. If USE_KEY_VAULT True and KEY_VAULT_URI present -> load Key Vault secrets once
      3. For each config variable, apply priority: KV > .env > warn and set to ""/None
      4. Convert numeric types where required
    """
    # Load Key Vault toggles from .env first
    USE_KEY_VAULT = os.getenv("USE_KEY_VAULT", "").lower() in ("true", "1", "yes")
    KEY_VAULT_URI = os.getenv("KEY_VAULT_URI", "")
    AZURE_USE_DEFAULT_CREDENTIAL = os.getenv("AZURE_USE_DEFAULT_CREDENTIAL", "").lower() in ("true", "1", "yes")

    # Set minimal KV related attributes on Config
    setattr(Config, "USE_KEY_VAULT", USE_KEY_VAULT)
    setattr(Config, "KEY_VAULT_URI", KEY_VAULT_URI)
    setattr(Config, "AZURE_USE_DEFAULT_CREDENTIAL", AZURE_USE_DEFAULT_CREDENTIAL)

    # If Key Vault is enabled and URI present, populate cache
    if USE_KEY_VAULT and KEY_VAULT_URI:
        try:
            Config._load_keyvault_secrets()
        except Exception as e:
            logger.warning("Failed to load Key Vault secrets: %s", e)

    # ---- Variables to load ----
    # Special-case Azure Search variables: ALWAYS read from .env (never Key Vault)
    AZURE_SEARCH_VARS = [
        "AZURE_SEARCH_ENDPOINT",
        "AZURE_SEARCH_API_KEY",
        "AZURE_SEARCH_INDEX_NAME",
    ]

    # Service principal vars to skip when using DefaultAzureCredential
    AZURE_SP_VARS = ["AZURE_TENANT_ID", "AZURE_CLIENT_ID", "AZURE_CLIENT_SECRET"]

    # All other variables required by agent / bundled modules.
    # Note: Do NOT add hardcoded fallback values here (except where explicitly allowed).
    CONFIG_VARIABLES = [
        # General
        "ENVIRONMENT",
        "VERSION",
        # Agent identity
        "AGENT_NAME",
        "AGENT_ID",
        "PROJECT_NAME",
        "PROJECT_ID",
        "SERVICE_NAME",
        "SERVICE_VERSION",
        # LLM / Model configuration (use provided defaults per build-time requirement)
        "MODEL_PROVIDER",  # Note: use env var MODEL_PROVIDER (platform uses this)
        "LLM_MODEL",
        "LLM_TEMPERATURE",
        "LLM_MAX_TOKENS",
        "AZURE_OPENAI_ENDPOINT",
        # API keys
        "OPENAI_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        # Content Safety
        "AZURE_CONTENT_SAFETY_ENDPOINT",
        "AZURE_CONTENT_SAFETY_KEY",
        "CONTENT_SAFETY_ENABLED",
        "CONTENT_SAFETY_SEVERITY_THRESHOLD",
        "CONTENT_SAFETY_CHECK_INPUT",
        "CONTENT_SAFETY_CHECK_OUTPUT",
        # Azure AI Search (always from .env)
        "AZURE_SEARCH_ENDPOINT",
        "AZURE_SEARCH_API_KEY",
        "AZURE_SEARCH_INDEX_NAME",
        # Observability DB (Azure SQL)
        "OBS_DATABASE_TYPE",
        "OBS_AZURE_SQL_SERVER",
        "OBS_AZURE_SQL_DATABASE",
        "OBS_AZURE_SQL_PORT",
        "OBS_AZURE_SQL_USERNAME",
        "OBS_AZURE_SQL_PASSWORD",
        "OBS_AZURE_SQL_SCHEMA",
        "OBS_AZURE_SQL_TRUST_SERVER_CERTIFICATE",
        # Observability / logging
        "LOG_LEVEL",
        # Domain / agent-specific extras
        "CACHE_TTL_SECONDS",
        "DEFAULT_PRECISION",
        "USER_PROMPT_TEMPLATE_ENHANCED",
        "USER_PROMPT_TEMPLATE_OUTPUT_FORMAT",
        "USER_PROMPT_TEMPLATE_FALLBACK_RESPONSE",
        "SELECTED_DOCUMENT_TITLES",
        "VALIDATION_CONFIG_PATH",
        "LLM_MODELS",
        # Azure Service Principal (used only if AZURE_USE_DEFAULT_CREDENTIAL is False)
        "AZURE_TENANT_ID",
        "AZURE_CLIENT_ID",
        "AZURE_CLIENT_SECRET",
        # Back-compat / observability expects settings instance attributes
        "SERVICE_VERSION",
    ]

    # Helper to determine KV vs env value
    def _get_value(var_name: str) -> Optional[str]:
        # Special handling: Azure Search vars always from .env
        if var_name in AZURE_SEARCH_VARS:
            return os.getenv(var_name, "")

        # Conditional skip: service principal vars when using DefaultAzureCredential
        if AZURE_USE_DEFAULT_CREDENTIAL and var_name in AZURE_SP_VARS:
            return None  # Skip entirely

        # Priority: Key Vault > .env
        if Config.USE_KEY_VAULT and var_name in getattr(Config, "_kv_secrets", {}):
            return Config._kv_secrets.get(var_name)

        # Else load from .env
        return os.getenv(var_name, "")

    # Iterate and set attributes
    for var in CONFIG_VARIABLES:
        # Skip AZURE_SP_VARS when using default credential
        if AZURE_USE_DEFAULT_CREDENTIAL and var in AZURE_SP_VARS:
            # Do not set these attrs at all (not needed)
            setattr(Config, var, "")
            continue

        # Special-case defaults for MODEL_PROVIDER and LLM_MODEL as required by build-time instruction.
        if var == "MODEL_PROVIDER":
            # Must use the provided default LLM provider value if not present in env/kv.
            # IMPORTANT: The default provider string is injected at build time below.
            value = os.getenv("MODEL_PROVIDER", "openai")
            # However ensure we honor Key Vault when USE_KEY_VAULT is enabled and secret exists
            if Config.USE_KEY_VAULT and "MODEL_PROVIDER" in Config._kv_secrets:
                value = Config._kv_secrets.get("MODEL_PROVIDER")
        elif var == "LLM_MODEL":
            value = os.getenv("LLM_MODEL", "gpt-5-mini")
            if Config.USE_KEY_VAULT and "LLM_MODEL" in Config._kv_secrets:
                value = Config._kv_secrets.get("LLM_MODEL")
        else:
            value = _get_value(var)

        # Normalize None to empty string for most string fields
        if value is None:
            # For optional fields we may set None; generally use empty string
            final_value = ""
        else:
            final_value = value

        # If value is empty or falsy, log a WARNING per policy and set to "" (or special-case)
        if final_value == "" or final_value is None:
            # Exception: OBS_AZURE_SQL_TRUST_SERVER_CERTIFICATE defaults to "yes"
            if var == "OBS_AZURE_SQL_TRUST_SERVER_CERTIFICATE":
                final_value = os.getenv("OBS_AZURE_SQL_TRUST_SERVER_CERTIFICATE", "yes")
            else:
                # Only log the standardized warning message referencing .env
                logging.warning("Configuration variable %s not found in .env file", var)
                # Keep final_value as empty string
                final_value = "" if final_value != None else ""

        # Type conversions for numeric fields
        if final_value != "" and var == "LLM_TEMPERATURE":
            try:
                final_value = float(final_value)
            except Exception:
                logging.warning("Invalid float value for %s: %s", var, final_value)
                final_value = ""
        elif final_value != "" and var == "LLM_MAX_TOKENS":
            try:
                final_value = int(final_value)
            except Exception:
                logging.warning("Invalid integer value for %s: %s", var, final_value)
                final_value = ""
        elif final_value != "" and var == "OBS_AZURE_SQL_PORT":
            try:
                final_value = int(final_value)
            except Exception:
                logging.warning("Invalid integer value for %s: %s", var, final_value)
                final_value = ""
        # SELECTED_DOCUMENT_TITLES could be a JSON list in env; try parse
        elif final_value != "" and var == "SELECTED_DOCUMENT_TITLES":
            try:
                # If someone set a JSON array in env, parse it; otherwise split comma
                parsed = json.loads(final_value)
                if isinstance(parsed, list):
                    final_value = parsed
                else:
                    raise ValueError("not a list")
            except Exception:
                # Fallback: treat as comma-separated list
                final_value = [s.strip() for s in str(final_value).split(",") if s.strip()] if final_value else []
        # LLM_MODELS may be JSON; leave raw string if parsing fails
        elif final_value != "" and var == "LLM_MODELS":
            try:
                parsed = json.loads(final_value)
                final_value = parsed
            except Exception:
                # leave as-is (observability code accepts falsy or list)
                pass

        # Finally set attribute on Config
        setattr(Config, var, final_value)

    # After all values set, ensure MODEL_PROVIDER and LLM_MODEL reflect the required defaults
    # Replace the placeholder defaults with injected values if necessary.
    # NOTE: The build-time requirement mandates these exact literal defaults:
    #   MODEL_PROVIDER default -> "openai" (injected)
    #   LLM_MODEL default -> "gpt-5-mini" (injected)
    # We already used those defaults above when env/kv missing.

    # Also expose API key env var names for convenience
    # (these attributes are populated above from KV or .env)
    setattr(Config, "OPENAI_API_KEY", getattr(Config, "OPENAI_API_KEY", ""))
    setattr(Config, "AZURE_OPENAI_API_KEY", getattr(Config, "AZURE_OPENAI_API_KEY", ""))
    setattr(Config, "ANTHROPIC_API_KEY", getattr(Config, "ANTHROPIC_API_KEY", ""))
    setattr(Config, "GOOGLE_API_KEY", getattr(Config, "GOOGLE_API_KEY", ""))

# Initialize configuration at module import time
_initialize_config()

# Backward compatibility settings instance
settings = Config()
