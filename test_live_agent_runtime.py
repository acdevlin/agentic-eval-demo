"""Test that are used by a *live* LLM runtime to test agent behavior."""

import pytest

# XXX DEBUG
import pprint

from conductor.ai.agents.runtime.config import AgentConfig
from conductor.ai.agents.runtime.runtime import AgentRuntime
from conductor.ai.agents.testing import (
    CorrectnessEval,
    EvalCase,
    validate_strategy,
)

from event_capturing_runtime import EventCapturingRuntime
from agents import billing_agent, support_agent, technical_agent, _PROMPT_REFUND


# Shared fixture
@pytest.fixture(scope="module")
def runtime():
    config = AgentConfig.from_env()
    # Force polling mode — SSE puts call IDs as tool names and omits
    # HANDOFF events.  Polling generates correct tool names, handoff
    # events, and guardrail events from Conductor task inspection.
    config.streaming_enabled = False
    rt = AgentRuntime(settings=config)
    yield rt
    rt.shutdown()


# Helper function to help determine why failures occurred during a test evaluation.
def process_eval_results(eval_result):
    assert (
        eval_result.all_passed
    ), f"{eval_result.fail_count}/{eval_result.total} eval(s) failed:\n" + "\n".join(
        f"  - {c.name}: {[ch.message for ch in c.checks if not ch.passed]}"
        for c in eval_result.failed_cases()
    )


class TestSupportAgent:
    """Confirms this agent hands off to sub-agents as expected."""

    def test_successful_refund_request(self, runtime):
        billing_eval = EvalCase(
            name="TestSupportAgent: test_successful_refund_request",
            agent=support_agent,
            prompt=_PROMPT_REFUND,
            expect_handoff_to="billing",
            expect_output_contains=["refund"],
        )
        eval = CorrectnessEval(EventCapturingRuntime(runtime))
        eval_result = eval.run([billing_eval])
        eval_result.print_summary()
        process_eval_results(eval_result)
        # Retrieve the EvalCaseResult from running our first (and only) EvalCase.
        case_result = eval_result.cases[0]
        # Next, retrieve the specific AgentResult so we can easily access Agent-level specifics.
        agent_result = case_result.result
        assert agent_result is not None
        # Now we can confirm that our agent used the HANDOFF strategy.
        validate_strategy(support_agent, agent_result)
        print(
            "Full run in the Orkes Conductor UI: "
            f"https://developer.orkescloud.com/agentExecutions/{agent_result.execution_id}"
        )


class TestBillingAgent:
    """Confirms this agent uses its associated tools as expected."""

    def test_successful_order_lookup(self, runtime):
        lookup_order_eval = EvalCase(
            name="TestBillingAgent: test_successful_order_lookup",
            agent=billing_agent,
            prompt="Give me the status of order 123",
            expect_tools=["lookup_order"],
            expect_tools_not_used=["process_refund"],
            expect_no_handoff_to=[support_agent, technical_agent],
            expect_output_contains=["123", "order"],
        )
        eval = CorrectnessEval(EventCapturingRuntime(runtime))
        eval_result = eval.run([lookup_order_eval])
        eval_result.print_summary()
        process_eval_results(eval_result)
        agent_result = eval_result.cases[0].result
        assert agent_result is not None
        print(
            "Full run in the Orkes Conductor UI: "
            f"https://developer.orkescloud.com/agentExecutions/{agent_result.execution_id}"
        )

    def test_successful_refund_request(self, runtime):
        lookup_order_eval = EvalCase(
            name="TestBillingAgent: test_successful_refund_request",
            agent=billing_agent,
            prompt="Process a refund for order 123",
            expect_tools=["process_refund"],
            expect_no_handoff_to=[support_agent, technical_agent],
            expect_output_contains=["123", "order", "refund"],
        )
        eval = CorrectnessEval(EventCapturingRuntime(runtime))
        eval_result = eval.run([lookup_order_eval])
        eval_result.print_summary()
        process_eval_results(eval_result)
        agent_result = eval_result.cases[0].result
        assert agent_result is not None
        print(
            "Full run in the Orkes Conductor UI: "
            f"https://developer.orkescloud.com/agentExecutions/{agent_result.execution_id}"
        )
