import pytest

from ai_review.services.agent.loop.schema import AgentAction, AgentStepSchema, AgentTraceSchema
from ai_review.services.agent.loop.schema import AgentLoopResultSchema
from ai_review.services.review.gateway.review_agent_llm_gateway import ReviewAgentLLMGateway
from ai_review.tests.fixtures.services.artifacts import FakeArtifactsService
from ai_review.tests.fixtures.services.cost import FakeCostService
from ai_review.tests.fixtures.services.review.gateway.review_agent_llm_gateway import FakeAgentLoopService
from ai_review.tests.fixtures.services.review.gateway.review_agent_llm_gateway import FakeFallbackReviewLLMGateway


@pytest.mark.asyncio
async def test_agent_gateway_returns_agent_result(
        review_agent_llm_gateway: ReviewAgentLLMGateway,
        fake_cost_service: FakeCostService,
        fake_artifacts_service: FakeArtifactsService,
        fake_agent_loop_service: FakeAgentLoopService,
        fake_fallback_review_llm_gateway: FakeFallbackReviewLLMGateway,
):
    fake_agent_loop_service.responses["run"] = AgentLoopResultSchema(
        final_text="AGENT_RESPONSE",
        stop_reason="final",
        traces=[
            AgentTraceSchema(
                step=AgentStepSchema(action=AgentAction.FINAL, content="step-one"),
                iteration=1,
                raw_output="raw-step-one",
                prompt_tokens=11,
                completion_tokens=7,
                total_tokens=18,
            ),
            AgentTraceSchema(
                step=AgentStepSchema(action=AgentAction.FINAL, content="step-two"),
                iteration=2,
                raw_output="raw-step-two",
                prompt_tokens=5,
                completion_tokens=3,
                total_tokens=8,
            ),
        ],
    )

    result = await review_agent_llm_gateway.ask("PROMPT", "SYSTEM_PROMPT")
    assert result == "AGENT_RESPONSE"
    assert any(call[0] == "run" for call in fake_agent_loop_service.calls)
    calculate_calls = [call for call in fake_cost_service.calls if call[0] == "calculate"]
    assert len(calculate_calls) == 1
    assert calculate_calls[0][1]["result"].prompt_tokens == 16
    assert calculate_calls[0][1]["result"].completion_tokens == 10
    assert any(call[0] == "save_llm" for call in fake_artifacts_service.calls)
    save_call = next(call for call in fake_artifacts_service.calls if call[0] == "save_llm")
    assert save_call[1]["cost_report"] is not None
    assert fake_fallback_review_llm_gateway.calls == []


@pytest.mark.asyncio
async def test_agent_gateway_falls_back_to_default_gateway_on_error(
        review_agent_llm_gateway: ReviewAgentLLMGateway,
        fake_agent_loop_service: FakeAgentLoopService,
        fake_fallback_review_llm_gateway: FakeFallbackReviewLLMGateway,
):
    fake_agent_loop_service.responses["raise"] = True
    fake_fallback_review_llm_gateway.responses["ask"] = "ONE_SHOT_RESPONSE"

    result = await review_agent_llm_gateway.ask("PROMPT", "SYSTEM_PROMPT")
    assert result == "ONE_SHOT_RESPONSE"
    assert any(call[0] == "ask" for call in fake_fallback_review_llm_gateway.calls)


@pytest.mark.asyncio
async def test_agent_gateway_calculates_zero_cost_for_missing_trace_tokens(
        review_agent_llm_gateway: ReviewAgentLLMGateway,
        fake_cost_service: FakeCostService,
        fake_agent_loop_service: FakeAgentLoopService,
):
    fake_agent_loop_service.responses["run"] = AgentLoopResultSchema(
        final_text="AGENT_RESPONSE",
        stop_reason="final",
        traces=[
            AgentTraceSchema(
                step=AgentStepSchema(action=AgentAction.FINAL, content="done"),
                iteration=1,
                raw_output="raw",
            ),
        ],
    )

    result = await review_agent_llm_gateway.ask("PROMPT", "SYSTEM_PROMPT")

    assert result == "AGENT_RESPONSE"
    calculate_calls = [call for call in fake_cost_service.calls if call[0] == "calculate"]
    assert len(calculate_calls) == 1
    assert calculate_calls[0][1]["result"].prompt_tokens == 0
    assert calculate_calls[0][1]["result"].completion_tokens == 0


@pytest.mark.asyncio
async def test_agent_gateway_propagates_cache_tokens_to_cost(
        review_agent_llm_gateway: ReviewAgentLLMGateway,
        fake_cost_service: FakeCostService,
        fake_agent_loop_service: FakeAgentLoopService,
):
    # Agent mode runs multiple LLM turns; cache_creation/cache_read tokens
    # must aggregate across traces and reach CalculateCostSchema so cache
    # pricing is applied. Pre-fix this was dropped silently.
    fake_agent_loop_service.responses["run"] = AgentLoopResultSchema(
        final_text="AGENT_RESPONSE",
        stop_reason="final",
        traces=[
            AgentTraceSchema(
                step=AgentStepSchema(action=AgentAction.FINAL, content="t1"),
                iteration=1,
                raw_output="raw-1",
                prompt_tokens=10,
                completion_tokens=4,
                cache_creation_tokens=1000,
                cache_read_tokens=0,
            ),
            AgentTraceSchema(
                step=AgentStepSchema(action=AgentAction.FINAL, content="t2"),
                iteration=2,
                raw_output="raw-2",
                prompt_tokens=12,
                completion_tokens=6,
                cache_creation_tokens=0,
                cache_read_tokens=900,
            ),
        ],
    )

    result = await review_agent_llm_gateway.ask("PROMPT", "SYSTEM_PROMPT")
    assert result == "AGENT_RESPONSE"
    calculate_calls = [call for call in fake_cost_service.calls if call[0] == "calculate"]
    assert len(calculate_calls) == 1
    schema = calculate_calls[0][1]["result"]
    assert schema.cache_creation_tokens == 1000
    assert schema.cache_read_tokens == 900
