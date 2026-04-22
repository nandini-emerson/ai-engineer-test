
import pytest
from pydantic import ValidationError
from agent import ComputeRequest

def test_compute_request_validation_rejects_empty_raw_expression():
    """Functional test: Constructing ComputeRequest with whitespace-only raw_expression raises ValidationError."""
    with pytest.raises(ValidationError) as excinfo:
        ComputeRequest(request_id="req-1", raw_expression="   ", precision=None, show_steps=False)
    # Extract textual messages from the ValidationError
    msgs = []
    try:
        msgs = [e.get("msg", "") for e in excinfo.value.errors()]
    except Exception:
        msgs = [str(excinfo.value)]
    # Assert the validator raised an error mentioning the raw_expression being empty
    assert any("raw_expression is empty" in m for m in msgs) or any(("raw_expression" in m and "empty" in m) for m in msgs), f"Unexpected validation messages: {msgs}"

def test_request_id_trims_and_enforces_presence():
    """Functional test: request_id validator trims whitespace and requires a non-empty value."""
    # Valid trimmed request_id
    req = ComputeRequest(request_id="  req-2  ", raw_expression="1+1", precision=None, show_steps=False)
    assert getattr(req, "request_id") == "req-2"

    # Whitespace-only request_id should raise ValidationError
    with pytest.raises(ValidationError) as excinfo2:
        ComputeRequest(request_id="   ", raw_expression="2+2", precision=None, show_steps=False)
    msgs2 = []
    try:
        msgs2 = [e.get("msg", "") for e in excinfo2.value.errors()]
    except Exception:
        msgs2 = [str(excinfo2.value)]
    assert any("request_id" in m and ("provided" in m or "must be" in m or "empty" in m) for m in msgs2) or any("request_id must be provided" in m for m in msgs2), f"Unexpected request_id validation messages: {msgs2}"
