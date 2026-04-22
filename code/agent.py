try:
    from observability.observability_wrapper import (
# AUTO-FIXED invalid syntax: import time as _time
        trace_agent, trace_step, trace_step_sync, trace_model_call, trace_tool_call,
    )
    from config import settings as _obs_settings
except ImportError:  # observability module not available (e.g. isolated test env)
    from contextlib import contextmanager as _obs_cm, asynccontextmanager as _obs_acm
    def trace_agent(*_a, **_kw):  # type: ignore[misc]
        def _deco(fn): return fn
        return _deco
    class _ObsHandle:
        output_summary = None
        def capture(self, *a, **kw): pass
    @_obs_acm
    async def trace_step(*_a, **_kw):  # type: ignore[misc]
        yield _ObsHandle()
    @_obs_cm
    def trace_step_sync(*_a, **_kw):  # type: ignore[misc]
        yield _ObsHandle()
    def trace_model_call(*_a, **_kw): pass  # type: ignore[misc]
    def trace_tool_call(*_a, **_kw): pass  # type: ignore[misc]
    class _ObsSettingsStub:
        AGENT_NAME: str = 'General Calculator Agent'
        PROJECT_NAME: str = '19test'
    _obs_settings = _ObsSettingsStub()


#!/usr/bin/env python3
"""
Calculator Agent - Orchestrator + Services

Implements a safe calculator agent per AGENT DESIGN:
- FastAPI HTTP ingress
- Orchestrator class (CalculatorAgentOrchestrator) that composes helper services
- Safe expression sanitization & limited parsing
- Deterministic numeric evaluation using sympy + Decimal rounding
- Optional LLM call for human-friendly explanations (Azure OpenAI async client)
- Observability lifespan init and trace_step usage (runtime provides trace_agent decorator)
- Pydantic v2 request validation with sanitization
- In-memory TTL cache (demo)
- Audit logging (best-effort)
"""

# SYNTAX-FIX: from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from contextlib import asynccontextmanager
from decimal import Decimal, getcontext, ROUND_HALF_EVEN
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import uvicorn
import openai
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, ValidationError, field_validator

# Local config import (lazy credential checks are enforced later)
from config import Config

# Guardrails decorator import (required by platform)
from modules.guardrails.content_safety_decorator import with_content_safety

# GUARDRAILS CONFIG (must be defined immediately after with_content_safety import)
GUARDRAILS_CONFIG = {
    'content_safety_enabled': True,
    'runtime_enabled': True,
    'content_safety_severity_threshold': 3,
    'check_toxicity': True,
    'check_jailbreak': True,
    'check_pii_input': False,
    'check_credentials_output': True,
    'check_output': True,
    'check_toxic_code_output': True,
    'sanitize_pii': False
}

# ---------------------------------------------------------------------------
# Module-level constants (INTERNAL CONFIG — not exposed via API)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT: str = getattr(Config, "USER_PROMPT_TEMPLATE_ENHANCED", None) or (
    "You are a professional expert calculator agent. Task: accept a user's textual math query or expression, "
    "validate and sanitize input, compute the result using safe internal math routines, and return the result. "
    "When requested or when calculation involves multiple steps, provide a clear step-by-step explanation. "
    "Always adhere to configured precision and unit conversion rules. If the input contains units, detect and perform "
    "conversions only when the target unit is specified or when the user requests conversion. Output format: produce "
    "a concise textual answer, optionally include 'steps' as a numbered list, and include a JSON block with keys: "
    "original_expression, parsed_expression, result, units (if applicable), precision_used, and explanation (optional). "
    "If the expression is invalid or contains unsupported operations, respond with a clear error code and suggested correction steps. "
    "Fallback: if the input cannot be safely computed, respond: 'Unable to compute: [reason]. Please simplify the expression or request a human review.'"
)

OUTPUT_FORMAT: str = getattr(Config, "USER_PROMPT_TEMPLATE_OUTPUT_FORMAT", None) or (
    "Primary: human-readable text with optional numbered step-by-step explanation. "
    "Secondary: structured JSON with fields original_expression, parsed_expression, result, units, precision_used, explanation."
)

FALLBACK_RESPONSE: str = getattr(Config, "USER_PROMPT_TEMPLATE_FALLBACK_RESPONSE", None) or (
    "Unable to compute: expression is invalid, ambiguous, or includes unsupported operations. "
    "Please simplify the expression, remove external data references, or request a human review."
)

SELECTED_DOCUMENT_TITLES: List[str] = getattr(Config, "SELECTED_DOCUMENT_TITLES", []) or []

VALIDATION_CONFIG_PATH: str = getattr(
    Config, "VALIDATION_CONFIG_PATH", str(Path(__file__).parent / "validation_config.json")
)

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logger = logging.getLogger("calculator_agent")
logging.basicConfig(level=Config.LOG_LEVEL if hasattr(Config, "LOG_LEVEL") else logging.INFO)

# ---------------------------------------------------------------------------
# FastAPI app + Observability lifespan (required)
# ---------------------------------------------------------------------------
_obs_startup_logger = logging.getLogger(__name__)

@asynccontextmanager
async def _obs_lifespan(application):
    """Initialise observability on startup, clean up on shutdown."""
    try:
        _obs_startup_logger.info("")
        _obs_startup_logger.info("========== Agent Configuration Summary ==========")
        _obs_startup_logger.info(f"Environment: {getattr(Config, 'ENVIRONMENT', 'N/A')}")
        _obs_startup_logger.info(f"Agent: {getattr(Config, 'AGENT_NAME', 'N/A')}")
        _obs_startup_logger.info(f"Project: {getattr(Config, 'PROJECT_NAME', 'N/A')}")
        _obs_startup_logger.info(f"LLM Provider: {getattr(Config, 'MODEL_PROVIDER', 'N/A')}")
        _obs_startup_logger.info(f"LLM Model: {getattr(Config, 'LLM_MODEL', 'N/A')}")
        _cs_endpoint = getattr(Config, 'AZURE_CONTENT_SAFETY_ENDPOINT', None)
        _cs_key = getattr(Config, 'AZURE_CONTENT_SAFETY_KEY', None)
        if _cs_endpoint and _cs_key:
            _obs_startup_logger.info("Content Safety: Enabled (Azure Content Safety)")
            _obs_startup_logger.info(f"Content Safety Endpoint: {_cs_endpoint}")
        else:
            _obs_startup_logger.info("Content Safety: Not Configured")
        _obs_startup_logger.info("Observability Database: Azure SQL")
        _obs_startup_logger.info(f"Database Server: {getattr(Config, 'OBS_AZURE_SQL_SERVER', 'N/A')}")
        _obs_startup_logger.info(f"Database Name: {getattr(Config, 'OBS_AZURE_SQL_DATABASE', 'N/A')}")
        _obs_startup_logger.info("===============================================")
        _obs_startup_logger.info("")
    except Exception as _e:
        _obs_startup_logger.warning('Config summary failed: %s', _e)

    # Guardrails configuration logging
    _obs_startup_logger.info("")
    _obs_startup_logger.info("========== Content Safety & Guardrails ==========")
    if GUARDRAILS_CONFIG.get('content_safety_enabled'):
        _obs_startup_logger.info("Content Safety: Enabled")
        _obs_startup_logger.info(f"  - Severity Threshold: {GUARDRAILS_CONFIG.get('content_safety_severity_threshold', 'N/A')}")
        _obs_startup_logger.info(f"  - Check Toxicity: {GUARDRAILS_CONFIG.get('check_toxicity', False)}")
        _obs_startup_logger.info(f"  - Check Jailbreak: {GUARDRAILS_CONFIG.get('check_jailbreak', False)}")
        _obs_startup_logger.info(f"  - Check PII Input: {GUARDRAILS_CONFIG.get('check_pii_input', False)}")
        _obs_startup_logger.info(f"  - Check Credentials Output: {GUARDRAILS_CONFIG.get('check_credentials_output', False)}")
    else:
        _obs_startup_logger.info("Content Safety: Disabled")
    _obs_startup_logger.info("===============================================")
    _obs_startup_logger.info("")

    _obs_startup_logger.info('========== Initializing Agent Services ==========')
    # Observability DB schema
    try:
        from observability.database.engine import create_obs_database_engine
        from observability.database.base import ObsBase
        import observability.database.models  # noqa: F401
        _obs_engine = create_obs_database_engine()
        ObsBase.metadata.create_all(bind=_obs_engine, checkfirst=True)
        _obs_startup_logger.info('✓ Observability database connected')
    except Exception as _e:
        _obs_startup_logger.warning('✗ Observability database connection failed (metrics will not be saved)')
    # OpenTelemetry tracer
    try:
        from observability.instrumentation import initialize_tracer
        _t = initialize_tracer()
        if _t is not None:
            _obs_startup_logger.info('✓ Telemetry monitoring enabled')
        else:
            _obs_startup_logger.warning('✗ Telemetry monitoring disabled')
    except Exception as _e:
        _obs_startup_logger.warning('✗ Telemetry monitoring failed to initialize')
    _obs_startup_logger.info('=================================================')
    _obs_startup_logger.info('')
    yield
    # Shutdown actions
    try:
        from observability.database.engine import close_obs_engine
        await close_obs_engine()
    except Exception:
        pass

app = FastAPI(
    title="General Calculator Agent",
    description="Calculator agent that evaluates sanitized numeric expressions and optionally explains steps.",
    version=getattr(Config, "SERVICE_VERSION", "1.0.0"),
    lifespan=_obs_lifespan
)

# ---------------------------------------------------------------------------
# Pydantic Request/Response models (Pydantic v2 style)
# ---------------------------------------------------------------------------

MAX_INPUT_CHARS = 50000

class ComputeRequest(BaseModel):
    request_id: str = Field(..., description="Unique request identifier")
    raw_expression: str = Field(..., description="The user's raw math expression or request")
    precision: Optional[int] = Field(None, description="Number of decimal places to return")
    show_steps: bool = Field(False, description="Whether to include step-by-step explanation")
    target_unit: Optional[str] = Field(None, description="Desired target unit (if conversion requested)")
    selected_document_titles: Optional[List[str]] = Field(None, description="Document titles for RAG filtering (internal)")

    @field_validator("raw_expression")
    @classmethod
    def validate_expression(cls, v: str) -> str:
        if v is None:
            raise ValueError("raw_expression is required")
        v = v.strip()
        if not v:
            raise ValueError("raw_expression is empty")
        if len(v) > MAX_INPUT_CHARS:
            raise ValueError(f"raw_expression exceeds maximum length ({MAX_INPUT_CHARS})")
        return v

    @field_validator("request_id")
    @classmethod
    def validate_request_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("request_id must be provided")
        return v.strip()

class ComputeResponse(BaseModel):
    success: bool
    human_text: str
    structured_json: Dict[str, Any]

# ---------------------------------------------------------------------------
# Simple in-memory TTL cache (Result-level) - demo only
# ---------------------------------------------------------------------------

class CacheClient:
    """
    Minimal in-memory TTL cache for demonstration. Not distributed; intended for unit tests.
    """
    def __init__(self):
        self._store: Dict[str, Tuple[float, Any]] = {}
        self._ttl_default = getattr(Config, "CACHE_TTL_SECONDS", 60)

    def _now(self) -> float:
        return time.time()

    def get(self, key: str) -> Optional[Any]:
        entry = self._store.get(key)
        if not entry:
            return None
        expires_at, value = entry
        if self._now() > expires_at:
            try:
                del self._store[key]
            except Exception:
                pass
            return None
        return value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        ttl = ttl if ttl is not None else self._ttl_default
        self._store[key] = (self._now() + float(ttl), value)

# ---------------------------------------------------------------------------
# Request Validator & Safe Expression Parser
# ---------------------------------------------------------------------------

ALLOWED_TOKEN_RE = re.compile(r"^[0-9\.\+\-\*\/\^\%\(\)\seE,A-Za-zµμ°]+$")  # allow letters for units
DISALLOWED_PATTERNS = [
    r"__", r"import\b", r"exec\b", r"eval\b", r"os\.", r"sys\.", r"subprocess\b", r"socket\b", r"open\("
]

class RequestValidator:
    """Sanitize raw input and enforce allowed token grammar (whitelist)."""

    def validate_and_sanitize(self, raw_expression: str) -> Tuple[str, bool, List[str]]:
        errors: List[str] = []
        expr = raw_expression.strip()
        # Normalize whitespace
        expr = re.sub(r"\s+", " ", expr)

        # Basic disallowed token checks
        for pat in DISALLOWED_PATTERNS:
            if re.search(pat, expr, re.IGNORECASE):
                errors.append("SECURITY_REJECTED")
                return expr, False, errors

        # Check allowed character set
        if not ALLOWED_TOKEN_RE.match(expr):
            errors.append("ERR_INVALID_EXPRESSION")
            return expr, False, errors

        # Normalize common caret operator '^' -> '**' for sympy compatibility later
        expr = expr.replace("^", "**")

        # Convert percent signs "50%" -> "(50/100)"
        expr = re.sub(r"(\d+(\.\d+)?)\s*%", r"(\1/100)", expr)

        return expr, True, []

class SafeExpressionParser:
    """
    Very small safe parser that:
    - Extracts a numeric-only expression for evaluation when possible
    - Detects simple unit patterns (e.g., '5 miles', '3 km')
    - Returns a 'parsed_expression' string and a structured representation used by evaluator
    """

    UNIT_TOKEN_RE = re.compile(r"(?P<number>\d+(\.\d+)?)\s*(?P<unit>[A-Za-zµμ°]+)")

    def parse(self, sanitized_expression: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Returns (parsed_dict, error_message). parsed_dict contains:
        - expression_for_eval (str) — expression to send to numeric evaluator (units stripped)
        - detected_units: List[dict] with number, unit, span_start, span_end
        - parsed_expression: canonical string describing parse
        """
        detected_units = []
        expr = sanitized_expression

        # Find simple "number unit" occurrences
        for m in self.UNIT_TOKEN_RE.finditer(expr):
            try:
                num = m.group("number")
                unit = m.group("unit")
                detected_units.append({"number": num, "unit": unit, "span": m.span()})
            except Exception:
                continue

        # Remove commas used in thousands (e.g., 1,000 -> 1000)
        expr_no_commas = re.sub(r"(?<=\d),(?=\d)", "", expr)

        parsed_expression = expr_no_commas
        parsed = {
            "expression_for_eval": parsed_expression,
            "detected_units": detected_units,
            "parsed_expression": parsed_expression,
        }
        return parsed, None

# ---------------------------------------------------------------------------
# Unit Conversion Service
# ---------------------------------------------------------------------------

class UnitConverter:
    """
    Minimal unit conversion table and converter for demonstration.
    Only a few units implemented. Real system should use comprehensive table.
    """

    _table = {
        ("mile", "mi", "kilometer", "km"): Decimal("1.609344"),
        ("km", "kilometer", "mi", "mile"): Decimal("0.62137119223733"),
        ("m", "meter", "km", "kilometer"): Decimal("0.001"),
        ("kg", "kilogram", "lb", "pound"): Decimal("2.2046226218"),
        ("lb", "pound", "kg", "kilogram"): Decimal("0.45359237"),
    }

    def _normalize_unit(self, u: str) -> str:
        return u.strip().lower()

    def convert(self, value: Decimal, source_unit: str, target_unit: str) -> Tuple[Optional[Decimal], List[str], Optional[str]]:
        s = self._normalize_unit(source_unit)
        t = self._normalize_unit(target_unit)
        # Try direct mapping
        for key in self._table:
            if s in key and t in key:
                # Determine direction: find factor from first pair to second pair mapping
                # Our _table keys are grouped; attempt to locate exact mapping
                try:
                    # If key like ("mile","mi","kilometer","km"), map mile->km factor
                    if key[0] in (s, t) and key[2] in (s, t):
                        # Find which way
                        if s in key[:2] and t in key[2:]:
                            factor = self._table[key]
                            converted = (value * factor).normalize()
                            steps = [f"Converted {value} {source_unit} -> {converted} {target_unit} using factor {factor}"]
                            return converted, steps, None
                        else:
                            # reverse direction
                            factor = self._table[key]
                            converted = (value / factor).normalize()
                            steps = [f"Converted {value} {source_unit} -> {converted} {target_unit} using factor 1/{factor}"]
                            return converted, steps, None
                except Exception as e:
                    return None, [], "ERR_UNSUPPORTED_OPERATION"
        # Try common abbreviations simple mapping (e.g., mi->km)
        # Fallback: unsupported unit
        return None, [], "ERR_UNSUPPORTED_OPERATION"

# ---------------------------------------------------------------------------
# Math Evaluator (deterministic)
# ---------------------------------------------------------------------------

# Local import of sympy is optional; use minimal subset to evaluate numeric expressions safely.
try:
    import sympy as _sympy  # type: ignore
    from sympy import sympify as _sympify
    SYMPY_AVAILABLE = True
except Exception:
    SYMPY_AVAILABLE = False

class MathEvaluator:
    """
    Evaluate numeric expressions safely. This evaluator does NOT execute arbitrary code.
    It relies on sympy when available and only after we enforce strict token validation.
    """

    def evaluate(self, expression: str, precision: int) -> Dict[str, Any]:
        """
        Returns dict: {result: Decimal, intermediate_values: [], steps: [], flags: {}}
        """
        if precision is None:
            precision = getattr(Config, "DEFAULT_PRECISION", 6)
        precision = int(precision)

        # Set decimal context precision slightly above requested to reduce rounding surprises
        getcontext().prec = max(precision + 6, 16)

        # Replace commas, ensure '**' is used for powers
        expr = expression.replace(",", "")

        if SYMPY_AVAILABLE:
            # Restrict sympify to prevent function calls: allow only Basic arithmetic
            try:
                # sympify with evaluate=True is okay because we've validated tokens earlier
                node = _sympify(expr, evaluate=True)
                # Evaluate numerically
                numeric = node.evalf(getcontext().prec)
                # Convert to Decimal safely using string representation
                result_decimal = Decimal(str(numeric))
                return {
                    "result": result_decimal,
                    "intermediate_values": [],
                    "steps": [],
                    "flags": {}
                }
            except Exception as e:
                logger.warning("Sympy evaluation failed: %s", e)
                raise ValueError("ERR_UNSUPPORTED_OPERATION")
        else:
            # Fallback: very limited safe eval using python's ast (only arithmetic)
            import ast, operator as _op

            # Define allowed operators
            operators = {
                ast.Add: _op.add,
                ast.Sub: _op.sub,
                ast.Mult: _op.mul,
                ast.Div: _op.truediv,
                ast.Pow: _op.pow,
                ast.USub: _op.neg,
                ast.UAdd: _op.pos,
                ast.Mod: _op.mod,
            }

            class SafeEval(ast.NodeVisitor):
                def visit(self, node):
                    if isinstance(node, ast.Expression):
                        return self.visit(node.body)
                    elif isinstance(node, ast.BinOp):
                        left = self.visit(node.left)
                        right = self.visit(node.right)
                        op = type(node.op)
                        if op not in operators:
                            raise ValueError("ERR_UNSUPPORTED_OPERATION")
                        return operators[op](left, right)
                    elif isinstance(node, ast.UnaryOp):
                        operand = self.visit(node.operand)
                        op = type(node.op)
                        if op not in operators:
                            raise ValueError("ERR_UNSUPPORTED_OPERATION")
                        return operators[op](operand)
                    elif isinstance(node, ast.Num):
                        return Decimal(str(node.n))
                    elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                        return Decimal(str(node.value))
                    else:
                        raise ValueError("ERR_UNSUPPORTED_OPERATION")

            try:
                tree = ast.parse(expr, mode='eval')
                val = SafeEval().visit(tree)
                return {"result": val, "intermediate_values": [], "steps": [], "flags": {}}
            except ValueError:
                raise
            except Exception as e:
                logger.warning("Fallback evaluator failed: %s", e)
                raise ValueError("ERR_UNSUPPORTED_OPERATION")

# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------

class Formatter:
    """Format Decimal results according to precision and rounding rules."""

    def format_result(self, raw_result: Decimal, precision: int, units: Optional[str],
                      show_steps: bool, parsed_expression: str, steps_text: Optional[str] = None) -> Dict[str, Any]:
        if precision is None:
            precision = getattr(Config, "DEFAULT_PRECISION", 6)
        precision = int(precision)
        # Apply bankers rounding (ROUND_HALF_EVEN) by default
        quantize_str = "1." + ("0" * precision) if precision > 0 else "1"
        if precision > 0:
            q = Decimal(1).scaleb(-precision)
        else:
            q = Decimal(1)
        rounded = raw_result.quantize(q, rounding=ROUND_HALF_EVEN)
        human_text = f"Result: {rounded}"
        if units:
            human_text = f"{human_text} {units}"
        structured = {
            "original_expression": parsed_expression,
            "parsed_expression": parsed_expression,
            "result": float(rounded) if abs(rounded) < Decimal("1e18") else str(rounded),
            "units": units or None,
            "precision_used": precision,
            "explanation": steps_text if show_steps else None
        }
        return {"human_text": human_text, "structured_json": structured}

# ---------------------------------------------------------------------------
# LLM Service Client (Azure OpenAI Async)
# ---------------------------------------------------------------------------

@with_content_safety(config=GUARDRAILS_CONFIG)
def get_llm_client():
    api_key = getattr(Config, "AZURE_OPENAI_API_KEY", None)
    if not api_key:
        raise ValueError("AZURE_OPENAI_API_KEY not configured")
    # Lazy instantiate AsyncAzureOpenAI client
    return openai.AsyncAzureOpenAI(
        api_key=api_key,
        api_version="2024-02-01",
        azure_endpoint=getattr(Config, "AZURE_OPENAI_ENDPOINT", None),
    )

class LLMServiceClient:
    """Call Azure OpenAI (AsyncAzureOpenAI) to generate explanations or polish steps."""

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def call_llm_for_explanation(
        self,
        sanitized_expression: str,
        parsed_expression: str,
        result: Decimal,
        steps: List[str],
        context_chunks: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Call the LLM to generate a human-friendly explanation.
        The system prompt MUST be SYSTEM_PROMPT appended with the OUTPUT_FORMAT instruction.
        """
        llm = get_llm_client()
        model = getattr(Config, "LLM_MODEL", None) or getattr(Config, "AZURE_OPENAI_DEPLOYMENT", None) or "gpt-4o"
        # Build user prompt from template
        user_prompt_template = getattr(Config, "USER_PROMPT_TEMPLATE", None) or (
            "User expression: {sanitized_expression}\nRequested precision: {precision}\nShow steps: {show_steps}\nUnits target: {target_unit}\nContext chunks: {rag_chunks_if_any}\nInstruction: Respond concisely, include steps if requested, and return a JSON block with original_expression, parsed_expression, result, units (if any), precision_used, and explanation (optional)."
        )
        user_prompt = user_prompt_template.format(
            sanitized_expression=sanitized_expression,
            precision=getattr(Config, "DEFAULT_PRECISION", None),
            show_steps=bool(steps),
            target_unit=None,
            rag_chunks_if_any=(context_chunks or [])
        )
        system_message = SYSTEM_PROMPT + "\n\nOutput Format: " + OUTPUT_FORMAT

        # Use Config.get_llm_kwargs() if available for dynamic param names
        llm_kwargs = getattr(Config, "get_llm_kwargs", lambda: {})()  # type: ignore
        try:
            # Chat completions create
            _obs_t0 = _time.time()
            resp = await llm.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_prompt},
                ],
                **llm_kwargs,
            )
            try:
                trace_model_call(
                    provider='azure',
                    model_name=(getattr(self, "model", None) or getattr(getattr(self, "config", None), "model", None) or "unknown"),
                    prompt_tokens=(getattr(getattr(resp, "usage", None), "prompt_tokens", 0) or 0),
                    completion_tokens=(getattr(getattr(resp, "usage", None), "completion_tokens", 0) or 0),
                    latency_ms=int((_time.time() - _obs_t0) * 1000),
                )
            except Exception:
                pass
            # Access response correctly
            content = ""
            try:
                content = resp.choices[0].message.content
            except Exception:
                # Fallback parsing
                content = getattr(resp, "text", "") or str(resp)
            return content
        except Exception as e:
            logger.warning("LLM call failed: %s", e)
            raise

# ---------------------------------------------------------------------------
# Audit Logger (best-effort)
# ---------------------------------------------------------------------------

class AuditLogger:
    """Persist sanitized request and results (best-effort)."""

    def log(self, request_id: str, sanitized_input: str, result_or_error: Dict[str, Any], metadata: Dict[str, Any]) -> Optional[str]:
        try:
            # Here we simply write a structured log. Production: persist to DB or secure log store.
            entry = {
                "request_id": request_id,
                "sanitized_input": sanitized_input,
                "result_or_error": result_or_error,
                "metadata": metadata,
                "timestamp": datetime.utcnow().isoformat()
            }
            logger.info("AUDIT_LOG: %s", json.dumps(entry, default=str))
            return f"log:{request_id}"
        except Exception as e:
            logger.warning("Audit logging failed: %s", e)
            return None

# ---------------------------------------------------------------------------
# Orchestrator (main agent class)
# ---------------------------------------------------------------------------

class CalculatorAgentOrchestrator:
    """
    Main orchestrator that composes validator, parser, evaluator, unit converter,
    formatter, cache, LLM client, and audit logger.
    """

    def __init__(self):
        self.validator = RequestValidator()
        self.parser = SafeExpressionParser()
        self.evaluator = MathEvaluator()
        self.unit_converter = UnitConverter()
        self.formatter = Formatter()
        self.llm = LLMServiceClient()
        self.cache = CacheClient()
        self.audit = AuditLogger()
        self.model = getattr(Config, "LLM_MODEL", "gpt-4o")

    @with_content_safety(config=GUARDRAILS_CONFIG)
    @trace_agent(agent_name=_obs_settings.AGENT_NAME, project_name=_obs_settings.PROJECT_NAME)
    async def process(self, request_id: str, raw_expression: str, precision: Optional[int],
                      show_steps: bool, target_unit: Optional[str], selected_document_titles: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Top-level entry point for agent execution. This is the method that the runtime
        will apply observability.trace_agent to.
        """
        # Observability: wrap logical stages with trace_step (runtime-provided)
        async with trace_step(
            "parse_input", step_type="parse",
            decision_summary="Sanitize and parse user input",
            # SYNTAX-FIX: output_fn=lambda r: f"valid={r.get('validation', False')}" if isinstance(r, dict) else str(r)
        ) as parse_step:
            # Validate & sanitize
            sanitized, valid, errors = self.validator.validate_and_sanitize(raw_expression)
            parse_step.capture({"sanitized": sanitized, "validation": valid, "errors": errors})
            if not valid:
                human = FALLBACK_RESPONSE
                resp = {"success": False, "human_text": human, "structured_json": {"error_code": errors or ["ERR_INVALID_EXPRESSION"]}}
                # Best-effort audit
                self.audit.log(request_id, sanitized, resp, {"stage": "validate"})
                return resp

            # Cache lookup
            cache_key = f"calc:{sanitized}:{precision}:{show_steps}:{target_unit}"
            cached = self.cache.get(cache_key)
            if cached:
                logger.info("Cache hit for request_id=%s", request_id)
                return {"success": True, "human_text": cached["human_text"], "structured_json": cached["structured_json"]}

            # Parse expression
            parsed, parse_err = self.parser.parse(sanitized)
            if parse_err:
                human = FALLBACK_RESPONSE
                resp = {"success": False, "human_text": human, "structured_json": {"error_code": ["ERR_INVALID_EXPRESSION"]}}
                self.audit.log(request_id, sanitized, resp, {"stage": "parse"})
                return resp

        # Evaluate (separate trace step)
        async with trace_step("evaluate", step_type="llm_call", decision_summary="Evaluate numeric expression", output_fn=lambda r: f"result={str(r.get('result')) if isinstance(r, dict) else '?'}") as eval_step:
            expr_for_eval = parsed.get("expression_for_eval")
            # If units detected and target_unit specified, we will extract numeric portion, convert after evaluation
            detected_units = parsed.get("detected_units", [])
            source_unit = None
            if detected_units:
                # Simplified: take first detected unit for conversion scenarios
                source_unit = detected_units[0].get("unit")

            try:
                eval_result = self.evaluator.evaluate(expr_for_eval, precision or getattr(Config, "DEFAULT_PRECISION", 6))
                eval_step.capture({"result": str(eval_result.get("result"))})
            except ValueError as e:
                human = FALLBACK_RESPONSE
                resp = {"success": False, "human_text": human, "structured_json": {"error_code": ["ERR_UNSUPPORTED_OPERATION"]}}
                self.audit.log(request_id, sanitized, resp, {"stage": "evaluate"})
                return resp

            raw_result: Decimal = eval_result["result"]

            conversion_steps_text = None
            final_units = None
            # Unit conversion if requested
            if target_unit and source_unit:
                converted_value, conv_steps, conv_err = self.unit_converter.convert(raw_result, source_unit, target_unit)
                if conv_err:
                    # Log and continue returning original value with error code
                    human = FALLBACK_RESPONSE
                    resp = {"success": False, "human_text": human, "structured_json": {"error_code": [conv_err]}}
                    self.audit.log(request_id, sanitized, resp, {"stage": "convert"})
                    return resp
                raw_result = converted_value
                conversion_steps_text = "; ".join(conv_steps)
                final_units = target_unit
            elif source_unit:
                final_units = source_unit

        # Format result
        async with trace_step("format_result", step_type="format", decision_summary="Apply precision and format", output_fn=lambda r: f"human_len={len(r.get('human_text','') if isinstance(r, dict) else 0)}") as fmt_step:
            try:
                formatted = self.formatter.format_result(raw_result, precision or getattr(Config, "DEFAULT_PRECISION", 6),
                                                         final_units, show_steps, parsed.get("parsed_expression"),
                                                         steps_text=conversion_steps_text)
                fmt_step.capture(formatted)
            except Exception as e:
                human = FALLBACK_RESPONSE
                resp = {"success": False, "human_text": human, "structured_json": {"error_code": ["ERR_PRECISION_LIMIT"]}}
                self.audit.log(request_id, sanitized, resp, {"stage": "format"})
                return resp

        # Optionally call LLM for explanation polishing
        llm_text = None
        if show_steps:
            async with trace_step("call_llm", step_type="llm_call", decision_summary="Call LLM for explanation", output_fn=lambda r: f"len={len(r) if r else 0}") as llm_step:
                try:
                    llm_text = await self.llm.call_llm_for_explanation(
                        sanitized_expression=sanitized,
                        parsed_expression=parsed.get("parsed_expression"),
                        result=raw_result,
                        steps=eval_result.get("steps", []),
                        context_chunks=[]
                    )
                    llm_step.capture(llm_text)
                except Exception as e:
                    # Fallback: use internal template explanation
                    llm_text = None
                    logger.warning("LLM explanation failed, falling back to template: %s", e)

        # Build response
        human_text = formatted["human_text"]
        structured = formatted["structured_json"]
        if llm_text:
            # Prefer LLM polished explanation if present
            structured["explanation"] = llm_text
            human_text = human_text + "\n\n" + (llm_text if isinstance(llm_text, str) else str(llm_text))

        result_payload = {"success": True, "human_text": human_text, "structured_json": structured}

        # Cache result
        try:
            self.cache.set(cache_key, {"human_text": human_text, "structured_json": structured})
        except Exception:
            logger.debug("Cache set failed for request_id=%s", request_id)

        # Audit log (best-effort)
        try:
            self.audit.log(request_id, sanitized, result_payload, {"stage": "complete"})
        except Exception:
            logger.debug("Audit log failed for request_id=%s", request_id)

        return result_payload

# Instantiate orchestrator (singleton)
agent = CalculatorAgentOrchestrator()

# ---------------------------------------------------------------------------
# FastAPI endpoints
# ---------------------------------------------------------------------------

@app.post("/compute", response_model=ComputeResponse)
async def compute_endpoint(req: ComputeRequest):
    """
    Compute endpoint. Only dynamic params from the design are accepted in the request body.
    SYSTEM_PROMPT and related internal constants are not exposed.
    """
    # Input sanitization has already been applied by Pydantic validators
    try:
        result = await agent.process(
            request_id=req.request_id,
            raw_expression=req.raw_expression,
            precision=req.precision,
            show_steps=req.show_steps,
            target_unit=req.target_unit,
            selected_document_titles=req.selected_document_titles
        )
        # Ensure success field present
        return JSONResponse(status_code=200 if result.get("success") else 400, content=result)
    except Exception as e:
        logger.exception("Unhandled error in compute_endpoint: %s", e)
        return JSONResponse(status_code=500, content={"success": False, "human_text": "Internal server error", "structured_json": {"error": "ERR_INTERNAL"}})

@app.get("/health")
async def health():
    return {"status": "ok"}

# JSON error handlers and validation helpers

@app.exception_handler(RequestValidationError)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # Pydantic v2 validation errors
    logger.warning("Request validation error: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"success": False, "human_text": "Invalid request JSON or parameters", "structured_json": {"error": "ERR_INVALID_REQUEST", "details": str(exc)}},
    )

@app.exception_handler(json.JSONDecodeError)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def json_decode_exception_handler(request: Request, exc: json.JSONDecodeError):
    logger.warning("Malformed JSON in request: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"success": False, "human_text": "Malformed JSON in request body. Check quotes, commas, and property names.", "structured_json": {"error": "ERR_MALFORMED_JSON"}}
    )

# ---------------------------------------------------------------------------
# Run server (async-compatible uvicorn pattern)
# ---------------------------------------------------------------------------



async def _run_agent():
    """Entrypoint: runs the agent with observability (trace collection only)."""

    # Unified logging config — routes uvicorn, agent, and observability through
    # the same handler so all telemetry appears in a single consistent stream.
    _LOG_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": "%(levelprefix)s %(name)s: %(message)s",
                "use_colors": None,
            },
            "access": {
                "()": "uvicorn.logging.AccessFormatter",
                "fmt": '%(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            },
            "access": {
                "formatter": "access",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            # uvicorn internals
            "uvicorn":        {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.error":  {"level": "INFO"},
            "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
            # agent application loggers
            "agent":          {"handlers": ["default"], "level": "INFO", "propagate": False},
            "__main__":       {"handlers": ["default"], "level": "INFO", "propagate": False},
            # observability / tracing namespace
            "observability": {"handlers": ["default"], "level": "INFO", "propagate": False},
            # config / settings namespace
            "config": {"handlers": ["default"], "level": "INFO", "propagate": False},
            # suppress noisy azure-sdk logs
            "azure":   {"handlers": ["default"], "level": "WARNING", "propagate": False},
            "urllib3": {"handlers": ["default"], "level": "WARNING", "propagate": False},
        },
    }

    config = uvicorn.Config(
        "agent:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="info",
        log_config=_LOG_CONFIG,
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    import asyncio as _asyncio
    _asyncio.run(_run_agent())