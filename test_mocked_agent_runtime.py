""" Tests that use a mocked runtime to confirm Agent correctness.

Note that we do *not* need to use the EventCapturingRuntime class for this test suite!
"""
from conductor.ai.agents.testing import (
    MockEvent,
    validate_strategy,
    expect,
    mock_run,
)

from agents import support_agent, _PROMPT_REFUND

class TestSupportAgent:
    """Confirm this agent hands off to sub-agents as expected."""

    def test_successful_order_lookup(self):
        result = mock_run(
            support_agent,
            "What is the status of order 123?",
            events=[
                MockEvent.handoff("billing"),
                MockEvent.tool_call("lookup_order", {"order_id": "123"}),
                MockEvent.tool_result(
                    "lookup_order", 
                    result={"order_id": "123", "status": "pending"}),
                MockEvent.done("Order #123 is currently 'pending' and has not shipped.")
            ],
            auto_execute_tools=False,
        )
        
        result.print_result()
        
        (expect(result)
            .completed()
            .used_tool("lookup_order")
            .did_not_use_tool("process_refund")
            .output_contains("pending")
            .output_contains("123")
            .no_errors())
        # Confirms that our agent did use Strategy.HANDOFF during mocked execution
        validate_strategy(support_agent, result)

    def test_successful_refund_request(self):
        result = mock_run(
            support_agent,
            _PROMPT_REFUND,
            events=[
                MockEvent.handoff("billing"),
                MockEvent.tool_call("lookup_order", {"order_id": "123"}),
                MockEvent.tool_result(
                    "lookup_order", 
                    result={"order_id": "123", "status": "shipped"}),
                MockEvent.tool_call("process_refund", args={"order_id": "123", "amount": 49.99}),
                MockEvent.tool_result("process_refund", result="Refund of $49.99 processed"),
                MockEvent.done("Your refund request for order #123 has been processed."),
            ],
            auto_execute_tools=False,
        )

        result.print_result()

        (expect(result)
            .completed()
            .agent_ran("billing")
            .tool_call_order(["lookup_order", "process_refund"])
            .did_not_use_tool("search_web")
            .output_contains("refund")
            .no_errors())
        # Confirms that our agent did use Strategy.HANDOFF during mocked execution
        validate_strategy(support_agent, result)
