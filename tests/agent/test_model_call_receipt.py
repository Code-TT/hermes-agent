from decimal import Decimal
from types import SimpleNamespace

from agent.turn_usage import build_model_call_receipt
from agent.usage_pricing import CanonicalUsage


def test_receipt_captures_route_tokens_and_cost_provenance():
    agent = SimpleNamespace(
        model="plan",
        provider="litellm",
        last_served_model="openrouter/deepseek/deepseek-v4-pro",
        last_served_api_base=None,
        last_served_response_cost=None,
        last_attempted_fallbacks=None,
    )
    response = SimpleNamespace(provider="OpenRouter", id="req-123")
    usage = CanonicalUsage(input_tokens=120, output_tokens=30, cache_read_tokens=40, cache_write_tokens=10, reasoning_tokens=4)
    cost = SimpleNamespace(amount_usd=Decimal("0.00125"), status="estimated", source="provider_models_api")

    receipt = build_model_call_receipt(agent, response, usage, cost, api_call_count=7)

    assert receipt == {
        "call_index": 7,
        "requested_model": "plan",
        "requested_provider": "litellm",
        "served_model": "openrouter/deepseek/deepseek-v4-pro",
        "served_provider": "OpenRouter",
        "response_id": "req-123",
        "upstream_api_base": None,
        "input_tokens": 120,
        "output_tokens": 30,
        "cache_read_tokens": 40,
        "cache_write_tokens": 10,
        "reasoning_tokens": 4,
        "total_tokens": 200,
        "cost_usd": 0.00125,
        "cost_status": "estimated",
        "cost_source": "provider_models_api",
        "attempted_fallbacks": None,
    }


def test_receipt_includes_mixture_of_agents_advisor_cost_in_turn_total():
    agent = SimpleNamespace(
        model="moa",
        provider="moa",
        last_served_model="openrouter/aggregator-model",
        last_served_api_base=None,
        last_served_response_cost=None,
        last_attempted_fallbacks=None,
    )
    response = SimpleNamespace(provider="OpenRouter", id="req-moa")
    usage = CanonicalUsage(input_tokens=100, output_tokens=20)
    aggregator_cost = SimpleNamespace(amount_usd=Decimal("0.001"), status="estimated", source="provider_models_api")

    receipt = build_model_call_receipt(
        agent, response, usage, aggregator_cost, api_call_count=2,
        additional_cost_usd=Decimal("0.002"),
    )

    assert receipt["requested_model"] == "moa"
    assert receipt["served_model"] == "openrouter/aggregator-model"
    assert receipt["cost_usd"] == 0.003
    assert receipt["cost_source"] == "provider_models_api+moa_advisors"


def test_receipt_keeps_unknown_route_and_cost_unknown_instead_of_guessing():
    agent = SimpleNamespace(
        model="plan",
        provider="litellm",
        last_served_model=None,
        last_served_api_base=None,
        last_served_response_cost=None,
        last_attempted_fallbacks=None,
    )
    response = SimpleNamespace(provider=None, id=None)
    usage = CanonicalUsage(input_tokens=8, output_tokens=2)
    cost = SimpleNamespace(amount_usd=None, status="unknown", source="none")

    receipt = build_model_call_receipt(agent, response, usage, cost, api_call_count=1)

    assert receipt["served_model"] is None
    assert receipt["served_provider"] is None
    assert receipt["cost_usd"] is None
    assert receipt["cost_status"] == "unknown"
