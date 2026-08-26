from conductor.ai.agents.testing import (
    MockEvent,
    assert_handoff_to,
    assert_output_contains,
    assert_tool_call_order,
    assert_tool_not_used,
    expect,
    mock_run,
)

from agents import support_agent, _PROMPT_REFUND

class TestSupportAgent:
    """Test that this agent hands off to sub-agents correctly and chains tools calls."""

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
                MockEvent.done("Your refund request for order #123 is has been processed."),
            ],
            auto_execute_tools=True,
        )
        result.print_result()
        expect(result).completed().no_errors()
        assert_handoff_to(result, "billing")
        assert_tool_call_order(result, ["lookup_order", "process_refund"])
        assert_tool_not_used(result, "search_web")
        assert_output_contains(result, "refund", case_sensitive=False)
