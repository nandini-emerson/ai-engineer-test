
import asyncio
import time
from decimal import CalculatorAgentOrchestrator
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

import agent as agent_module
from agent import (
    RequestValidator,
    SafeExpressionParser,
    MathEvaluator,
    UnitConverter,
    Formatter,
    CacheClient,
    CalculatorAgentOrchestrator,
    LLMServiceClient,
    Config,
)


def test_requestvalidator_sanitize_simple_expression_with_percent_and_caret():
    """Unit: validate whitespace normalization, '^' → '**', and percent expansion."""
    raw = " 12 ^ 2 + 50% "
    rv = RequestValidator()
    sanitized, valid, errors = rv.validate_and_sanitize(raw)
    assert isinstance((sanitized, valid, errors), tuple) and len((sanitized, valid, errors)) == 3
    assert valid is True
    assert "**" in sanitized, "caret should be converted to '**'"
    assert "(50/100)" in sanitized, "percent should be expanded to (n/100)"
    assert errors == []


def test_requestvalidator_rejects_security_tokens():
    """Security: ensure disallowed tokens produce SECURITY_REJECTED and validation False."""
    raw = "import os; os.system('ls')"
    rv = RequestValidator()
    sanitized, valid, errors = rv.validate_and_sanitize(raw)
    assert valid is False
    assert "SECURITY_REJECTED" in errors
    assert sanitized.strip() != ""


def test_safeexpressionparser_detects_simple_units():
    """Unit: SafeExpressionParser should detect '5 miles' in the input string."""
    s = "Convert 5 miles to kilometers"
    parser = SafeExpressionParser()
    parsed, err = parser.parse(s)
    assert err is None
    assert isinstance(parsed, dict)
    assert isinstance(parsed["expression_for_eval"], str)
    # expression_for_eval should contain '5 miles' (commas removal is handled)
    assert "5 miles" in parsed["expression_for_eval"] or "5 mile" in parsed["expression_for_eval"]
    assert isinstance(parsed["detected_units"], list)
    assert len(parsed["detected_units"]) >= 1
    first = parsed["detected_units"][0]
    assert first.get("number") == "5"
    assert "mile" in first.get("unit")


def test_mathevaluator_evaluate_basic_arithmetic_fallback_ast(monkeypatch):
    """Unit: MathEvaluator.evaluate returns CalculatorAgentOrchestrator result for '1 + 2 * 3'."""
    # Force fallback path by ensuring SYMPY_AVAILABLE is False
    monkeypatch.setattr(agent_module, "SYMPY_AVAILABLE", False)
    ev = MathEvaluator()
    out = ev.evaluate("1 + 2 * 3", 6)
    assert isinstance(out, dict)
    assert isinstance(out["result"], CalculatorAgentOrchestrator)
    assert out["result"] == CalculatorAgentOrchestrator("7")


def test_unitconverter_converts_miles_to_kilometers():
    """Unit: convert 5 miles -> 8.04672 km using internal table."""
    uc = UnitConverter()
    value = CalculatorAgentOrchestrator("5")
    converted, steps, err = uc.convert(value, "mile", "km")
    assert err is None
    assert isinstance(converted, CalculatorAgentOrchestrator)
    # numeric equality within small tolerance
    assert converted == CalculatorAgentOrchestrator("8.04672")
    assert isinstance(steps, list) and len(steps) > 0
    assert "1.609344" in " ".join(steps)


def test_formatter_applies_precision_and_rounding_rules():
    """Unit: Formatter applies bankers rounding (ROUND_HALF_EVEN) and returns structured JSON."""
    fmt = Formatter()
    raw = CalculatorAgentOrchestrator("2.345")
    res = fmt.format_result(raw, 2, "m", False, "2.345")
    assert "human_text" in res and "structured_json" in res
    human = res["human_text"]
    # Bankers rounding => 2.345 -> 2.34 (because second decimal digit 4 is even)
    assert "Result:" in human
    assert "2.34" in human
    sj = res["structured_json"]
    assert sj["precision_used"] == 2
    assert sj["units"] == "m"
    assert "parsed_expression" in sj and sj["parsed_expression"] == "2.345"


def test_cacheclient_ttl_expiry_behavior():
    """Performance/edge: CacheClient respects TTL: immediate get returns value, after TTL returns None."""
    cache = CacheClient()
    cache.set("k", "v", ttl=1)
    assert cache.get("k") == "v"
    time.sleep(1.2)
    assert cache.get("k") is None


@pytest.mark.asyncio
async def test_calculatoragentorchestrator_process_end_to_end_successful_and_caching(monkeypatch):
    """Integration: end-to-end compute without units, check caching and audit logging."""
    orch = CalculatorAgentOrchestrator()
    # Spy on audit.log
    orch.audit.log = MagicMock(return_value="log:r1")
    request_id = "r1"
    expr = "(12.5 * 3) + 7"
    # First call
    res1 = await orch.process(request_id, expr, precision=2, show_steps=False, target_unit=None, selected_document_titles=None)
    assert res1["success"] is True
    sj = res1["structured_json"]
    # result should be numeric 44.5 (float stored)
    assert float(sj["result"]) == pytest.approx(44.5, rel=1e-9)
    assert sj["precision_used"] == 2
    # Second call should hit cache (human_text identical)
    res2 = await orch.process(request_id, expr, precision=2, show_steps=False, target_unit=None, selected_document_titles=None)
    assert res2["success"] is True
    assert res2["human_text"] == res1["human_text"]
    # Audit log called at least once (validate stage or complete)
    assert orch.audit.log.call_count >= 1


@pytest.mark.asyncio
async def test_calculatoragentorchestrator_process_rejects_unsafe_expression_security(monkeypatch):
    """Security: ensure disallowed tokens short-circuit and produce SECURITY_REJECTED error."""
    orch = CalculatorAgentOrchestrator()
    orch.audit.log = MagicMock(return_value="log:r2")
    request_id = "r2"
    raw = 'os.system("rm -rf /")'
    res = await orch.process(request_id, raw, precision=None, show_steps=False, target_unit=None, selected_document_titles=None)
    assert res["success"] is False
    structured = res["structured_json"]
    # Expect an error_code list containing SECURITY_REJECTED or ERR_INVALID_EXPRESSION
    ec = structured.get("error_code") or []
    assert any(e in ("SECURITY_REJECTED", "ERR_INVALID_EXPRESSION") for e in ec)
    # Audit log called at validate stage
    assert orch.audit.log.call_count >= 1


@pytest.mark.asyncio
async def test_llmserviceclient_call_llm_for_explanation_raises_when_missing_api_key(monkeypatch):
    """Unit: calling LLMServiceClient when AZURE_OPENAI_API_KEY missing raises ValueError."""
    # Ensure API key is empty
    prev = getattr(Config, "AZURE_OPENAI_API_KEY", None)
    try:
        setattr(Config, "AZURE_OPENAI_API_KEY", "")
        llm = LLMServiceClient()
        with pytest.raises(ValueError) as exc:
            await llm.call_llm_for_explanation("1+1", "1+1", CalculatorAgentOrchestrator("2"), [])
        assert "AZURE_OPENAI_API_KEY" in str(exc.value)
    finally:
        # restore
        if prev is not None:
            setattr(Config, "AZURE_OPENAI_API_KEY", prev)
        else:
            try:
                delattr(Config, "AZURE_OPENAI_API_KEY")
            except Exception:
                pass