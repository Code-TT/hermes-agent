"""Per-response *served model* capture for routing proxies (#54864).

A LiteLLM-style proxy answers with the configured alias in the body's ``model`` field and puts
the deployment it really routed to in a response header (``x-litellm-model-id``, else the
upstream ``x-litellm-model-api-base``). The OpenAI SDK's parsed objects drop headers, so the
capture rides an ``httpx`` response hook on the client Hermes builds for the agent; it stores
the header onto ``agent.last_served_model`` (``None`` when the response carried none, so a
value never outlives the request that produced it). Consumers: ``agent/turn_finalizer.py``
(result ``served_model`` / ``requested_model``) and the opt-in gateway footer field
``served_model`` (``gateway/runtime_footer.py``). Fail-open everywhere.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

SERVED_MODEL_HEADERS: tuple[str, ...] = ("x-litellm-model-id", "x-litellm-model-api-base")
_HOOK_MARK = "_hermes_served_model_hook"


def served_model_metadata_from_headers(headers: Any) -> dict[str, Any]:
    """Extract routing metadata without confusing a proxy alias with a real route."""
    if headers is None or not hasattr(headers, "get"):
        return {"model": None, "api_base": None, "response_cost": None, "attempted_fallbacks": None}
    model = headers.get("x-litellm-model-id")
    api_base = headers.get("x-litellm-model-api-base")
    raw_cost = headers.get("x-litellm-response-cost")
    try:
        cost = float(raw_cost) if raw_cost not in (None, "") else None
        if cost is not None and (cost < 0 or cost != cost or cost == float("inf")):
            cost = None
    except (TypeError, ValueError):
        cost = None
    raw_fallbacks = headers.get("x-litellm-attempted-fallbacks")
    try:
        attempted_fallbacks = int(raw_fallbacks) if raw_fallbacks not in (None, "") else None
        if attempted_fallbacks is not None and attempted_fallbacks < 0:
            attempted_fallbacks = None
    except (TypeError, ValueError):
        attempted_fallbacks = None
    return {
        "model": str(model).strip() if model not in (None, "") else None,
        "api_base": str(api_base).strip() if api_base not in (None, "") else None,
        "response_cost": cost,
        "attempted_fallbacks": attempted_fallbacks,
    }


def served_model_from_headers(headers: Any) -> Optional[str]:
    """Return LiteLLM deployment id, or its API base if no id was supplied."""
    metadata = served_model_metadata_from_headers(headers)
    return metadata["model"] or metadata["api_base"]


def install_served_model_capture(agent: Any, client: Any) -> None:
    """Register the response hook on *client*'s ``httpx`` transport (idempotent per client)."""
    http_client = getattr(client, "_client", None)
    hooks = getattr(http_client, "event_hooks", None)
    if not isinstance(hooks, dict):
        return
    if any(getattr(h, _HOOK_MARK, False) for h in hooks.get("response", ())):
        return

    def _on_response(response: Any) -> None:
        try:
            # Clear before reading each successful response so route headers never leak
            # from a previous call when this response is missing a LiteLLM field.
            if 200 <= int(getattr(response, "status_code", 0)) < 300:
                metadata = served_model_metadata_from_headers(getattr(response, "headers", None))
                agent.last_served_model = metadata["model"]
                agent.last_served_api_base = metadata["api_base"]
                agent.last_served_response_cost = metadata["response_cost"]
                agent.last_attempted_fallbacks = metadata["attempted_fallbacks"]
        except Exception:
            logger.debug("served-model header capture skipped", exc_info=True)

    setattr(_on_response, _HOOK_MARK, True)
    try:
        # httpx copies on assignment; rebuild the mapping instead of mutating the live list.
        http_client.event_hooks = {**hooks, "response": [*hooks.get("response", ()), _on_response]}
    except Exception:
        logger.debug("served-model hook install skipped", exc_info=True)


def result_model_fields(agent: Any) -> dict[str, Optional[str]]:
    """``requested_model`` / ``served_model`` for the turn result: the proxy header when the
    last response carried one, else Hermes' own fallback route (primary → active model)."""
    served = getattr(agent, "last_served_model", None)
    requested = agent.model
    if not served and getattr(agent, "_fallback_activated", False):
        primary = str((getattr(agent, "_primary_runtime", None) or {}).get("model") or "").strip()
        if primary and primary != agent.model:
            requested, served = primary, agent.model
    return {"requested_model": requested, "served_model": served or None}
