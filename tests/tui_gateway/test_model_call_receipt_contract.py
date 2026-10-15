from tui_gateway.contracts.common import ModelCallReceipt, Usage


def test_latest_model_call_receipt_round_trips_on_usage_contract():
    receipt = ModelCallReceipt(
        call_index=3,
        requested_model="plan",
        requested_provider="litellm",
        served_model="deepseek/deepseek-v4-pro",
        served_provider="OpenRouter",
        response_id="req-abc",
        upstream_api_base="https://openrouter.ai/api/v1",
        input_tokens=900,
        output_tokens=55,
        cache_read_tokens=100,
        cache_write_tokens=10,
        reasoning_tokens=5,
        total_tokens=1065,
        cost_usd=0.001,
        cost_status="estimated",
        cost_source="litellm_response_header",
        attempted_fallbacks=1,
    )
    usage = Usage(model="plan", provider="litellm", latest_model_call=receipt)
    dumped = usage.model_dump()
    parsed = Usage.model_validate(dumped)
    assert parsed.latest_model_call.served_model == "deepseek/deepseek-v4-pro"
    assert parsed.latest_model_call.attempted_fallbacks == 1
    assert parsed.latest_model_call.cost_status == "estimated"
