General Calculator Agent
=======================

What it does
-----------
General Calculator Agent evaluates sanitized numeric and simple algebraic expressions, optionally performs unit conversions, and (when requested) produces step-by-step explanations. It exposes a small FastAPI HTTP API returning both human-readable text and a structured JSON block with original_expression, parsed_expression, result, units, precision_used and optional explanation.

Quick start
-----------
1. Clone and enter the project:
   - git clone <repo>
   - cd <repo>

2. Install dependencies:
   - pip install -r requirements.txt
   - For tests: pip install pytest pytest-asyncio

3. Configure environment:
   - Copy .env.example to .env and fill required values.

4. Run the agent locally:
   - python -m uvicorn agent:app --host 0.0.0.0 --port 8080
   - (Or use the provided Dockerfile to build and run the container.)

Environment variables
---------------------
Populate these (from .env.example). Core vars the agent uses:
- MODEL_PROVIDER (openai / azure / etc.)
- LLM_MODEL (e.g., gpt-5-mini)
- OPENAI_API_KEY
- AZURE_OPENAI_API_KEY
- AZURE_OPENAI_ENDPOINT
- LLM_TEMPERATURE
- LLM_MAX_TOKENS
- USER_PROMPT_TEMPLATE_ENHANCED
- USER_PROMPT_TEMPLATE_OUTPUT_FORMAT
- USER_PROMPT_TEMPLATE_FALLBACK_RESPONSE

Content safety / key vault / observability (optional but used if enabled):
- USE_KEY_VAULT, KEY_VAULT_URI, AZURE_USE_DEFAULT_CREDENTIAL
- AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET
- AZURE_CONTENT_SAFETY_ENDPOINT, AZURE_CONTENT_SAFETY_KEY
- AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_API_KEY, AZURE_SEARCH_INDEX_NAME
- OBS_DATABASE_TYPE, OBS_AZURE_SQL_SERVER, OBS_AZURE_SQL_DATABASE
- OBS_AZURE_SQL_PORT, OBS_AZURE_SQL_USERNAME, OBS_AZURE_SQL_PASSWORD, OBS_AZURE_SQL_SCHEMA

Agent runtime / behavior:
- DEFAULT_PRECISION (default numeric precision)
- CACHE_TTL_SECONDS (result cache TTL)
- VALIDATION_CONFIG_PATH
- LOG_LEVEL

API endpoints
-------------
- POST /compute
  - Description: Evaluate a sanitized math expression (and optionally convert units and show steps).
  - Request body (JSON):
    - request_id: string (required) — unique request id
    - raw_expression: string (required) — user expression (e.g., "(12.5 * 3) + 7")
    - precision: integer (optional) — decimal places to use
    - show_steps: boolean (optional) — include step-by-step explanation
    - target_unit: string (optional) — convert result to this unit
    - selected_document_titles: list (optional) — RAG filter (if RAG enabled)
  - Response (200 on success / 400 on validation):
    - success: bool
    - human_text: string (concise human-readable answer)
    - structured_json: object { original_expression, parsed_expression, result, units, precision_used, explanation (optional) }

- GET /health
  - Description: Simple liveness/health check returning {"status": "ok"}.

How to run tests
----------------
1. Install test deps:
   - pip install pytest pytest-asyncio

2. Run tests:
   - pytest -q

Notes and tips
--------------
- Never expose API keys in source or logs. Use .env or a vault.
- If using Azure Key Vault or Azure OpenAI, ensure corresponding environment variables and credentials are set.
- The agent uses a safe, whitelist-style parser and will reject inputs containing disallowed tokens (e.g., import, eval, os.*). Error responses include standardized error codes (ERR_INVALID_EXPRESSION, ERR_UNSUPPORTED_OPERATION, SECURITY_REJECTED, etc.).