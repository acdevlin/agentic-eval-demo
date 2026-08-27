"""Tests that use a mocked runtime to confirm Agent correctness.

This is used primarily to assert on routing and tool selection without incurring a token cost.
Note that we do *not* need to use the EventCapturingRuntime class here since the runtime is mocked.
"""

import pytest
from pprint import pprint

from conductor.ai.agents.testing import (
    MockEvent,
    StrategyViolation,
    assert_handoff_to,
    expect,
    mock_run,
    validate_strategy,
    assert_tool_used,
    assert_tool_called_with,
    assert_agent_ran,
)

from agents import (
    support_agent,
    summary_agent,
    editor_agent,
    billing_agent,
    technical_agent,
    _PROMPT_REFUND,
)


class TestSupportAgent:
    """Confirms this agent hands off to sub-agents as expected."""

    def test_only_one_specialist_runs(self):
        """Support agent should only pick a single specialist for handoff."""
        result = mock_run(
            support_agent,
            "What is the maximum flight speed velocity of an unladen swallow?",
            events=[
                MockEvent.thinking("Deciding which specialist to send request to..."),
                MockEvent.handoff("technical"),
                MockEvent.tool_call(
                    "search_web",
                    {"query": "Maximum flight speed velocity of an unladen swallow"},
                ),
                MockEvent.done("What do you mean - African, or European swallow?"),
            ],
            # Intentionally don't mock tool call results - we want to test the real tool code.
            auto_execute_tools=True,
        )

        result.print_result()
        # XXX DEBUG
        pprint(result)

        assert_handoff_to(result, "technical")
        # Expect this to be thrown since support should have only handed off to 'technical' agent.
        with pytest.raises(AssertionError):
            assert_handoff_to(result, "billing")

    def test_unknown_tool_name_throws_error(self):
        """Support agent telling specialist to use an unknown tool name throws an error."""
        result = mock_run(
            support_agent,
            "Use the 'write_pseudocode' function to explain Dijkstra's Algorithm",
            events=[
                MockEvent.handoff("technical"),
                MockEvent.tool_call(
                    "write_pseudocode", {"input": "Dijkstra's Algorithm"}
                ),
                MockEvent.error("Tool 'write_pseudocode' does not exist!"),
            ],
            auto_execute_tools=True,
        )

        result.print_result()
        # XXX DEBUG
        pprint(result)

        (
            expect(result)
            .failed()
            .agent_ran("technical")
            # Note that even though this tool doesn't exist it *is* considered to have executed!
            .used_tool("write_pseudocode")
        )

    def test_successful_order_lookup(self):
        """Basic test case that uses 1 tool through the billing agent."""
        result = mock_run(
            support_agent,
            "What is the status of order 123?",
            events=[
                MockEvent.handoff("billing"),
                MockEvent.tool_call("lookup_order", {"order_id": "123"}),
                MockEvent.tool_result(
                    "lookup_order", result={"order_id": "123", "status": "pending"}
                ),
                MockEvent.done(
                    "Order #123 is currently 'pending' and has not shipped."
                ),
            ],
            auto_execute_tools=False,
        )

        result.print_result()

        (
            expect(result)
            .completed()
            .used_tool("lookup_order")
            .did_not_use_tool("process_refund")
            .output_contains("pending")
            .output_contains("123")
            .no_errors()
        )
        # Confirms that our agent did use Strategy.HANDOFF during mocked execution
        validate_strategy(support_agent, result)

    def test_successful_refund_request(self):
        """Compound test case with 2 tool uses through the billing agent."""
        result = mock_run(
            support_agent,
            _PROMPT_REFUND,
            events=[
                MockEvent.handoff("billing"),
                MockEvent.tool_call("lookup_order", {"order_id": "123"}),
                MockEvent.tool_result(
                    "lookup_order", result={"order_id": "123", "status": "shipped"}
                ),
                MockEvent.tool_call(
                    "process_refund", args={"order_id": "123", "amount": 49.99}
                ),
                MockEvent.tool_result(
                    "process_refund", result="Refund of $49.99 processed"
                ),
                MockEvent.done(
                    "Your refund request for order #123 has been processed."
                ),
            ],
            auto_execute_tools=False,
        )

        result.print_result()

        (
            expect(result)
            .completed()
            .agent_ran("billing")
            .tool_call_order(["lookup_order", "process_refund"])
            .did_not_use_tool("search_web")
            .output_contains("refund")
            .no_errors()
        )
        # Confirms that our agent did use Strategy.HANDOFF during mocked execution
        validate_strategy(support_agent, result)

    def test_successful_calculation(self):
        """Basic use case with handoff to technical agent."""
        result = mock_run(
            support_agent,
            "What is 5 times 3?",
            events=[
                MockEvent.handoff("technical"),
                MockEvent.tool_call("calculate", {"expression": "5 * 3"}),
                MockEvent.tool_result("calculate", result={"result": "15"}),
                MockEvent.done("5 times 3 is 15"),
            ],
            auto_execute_tools=False,
        )

        result.print_result()

        (
            expect(result)
            .completed()
            .agent_ran("technical")
            .used_tool("calculate")
            .did_not_use_tool("search_web")
            .output_contains("15")
            .no_errors()
        )
        # Confirms that our agent did use Strategy.HANDOFF during mocked execution
        validate_strategy(support_agent, result)


class TestSequentialPipeline:
    content_pipeline = technical_agent >> summary_agent >> editor_agent

    def test_agent_order(self):
        """Verifies that the agents run in the specified order."""
        result = mock_run(
            self.content_pipeline,
            "Search for articles about AI safety, then write a summary about their contents.",
            events=[
                MockEvent.handoff("technical"),
                MockEvent.tool_call("search_web", args={"query": "AI safety"}),
                MockEvent.tool_result(
                    "search_web", result="AI safety research focuses on..."
                ),
                MockEvent.handoff("summary"),
                MockEvent.handoff("editor"),
                MockEvent.done("Summary of AI Safety: Ensuring Beneficial AI\n\n..."),
            ],
            auto_execute_tools=False,
        )

        result.print_result()

        # Confirm that specified agents ran
        assert_agent_ran(result, "technical")
        assert_agent_ran(result, "summary")
        assert_agent_ran(result, "editor")
        # Confirm that specified tools were used
        assert_tool_used(result, "search_web")
        assert_tool_called_with(result, "search_web", args={"query": "AI safety"})

    def test_skipped_agent_throws_error(self):
        """Confirms that if an agent is skipped, an error is thrown."""
        result = mock_run(
            self.content_pipeline,
            "Write about agentic AI",
            events=[
                MockEvent.handoff("technical"),
                # "summary" handoff is intentionally missing to trigger violations
                MockEvent.handoff("editor"),
                MockEvent.done("Incomplete article"),
            ],
        )

        result.print_result()

        # These two agents DID run
        assert_agent_ran(result, "technical")
        assert_agent_ran(result, "editor")

        # This agent DID NOT run
        with pytest.raises(AssertionError, match="summary"):
            assert_agent_ran(result, "summary")

        # Verify that the "summary" agent was skipped and caused a Strategy error
        with pytest.raises(StrategyViolation, match="skipped"):
            validate_strategy(self.content_pipeline, result)
