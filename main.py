from conductor.ai.agents import AgentRuntime
from conductor.ai.agents.testing import (
    MockEvent,
    assert_handoff_to,
    assert_output_contains,
    assert_tool_call_order,
    assert_tool_not_used,
    mock_run,
)
from conductor.client.configuration.configuration import Configuration
from conductor.client.configuration.settings.authentication_settings import AuthenticationSettings
import os
import argparse

from agents import support_agent, billing_agent, technical_agent

_PROMPT = "I need a refund for order 123."

def main():
    """
    The runtime is NOT mocked - all LLM calls are actually evaluated.
    """
    parser = argparse.ArgumentParser(
        description="Showcases a sample of handing off execution between 3 agents."
    )
    parser.add_argument(
        "--mock-eval",
        action="store_true",
        default=False,
        help="Evaluates routing and tool behavior without an LLM call."
    )
    args = parser.parse_args()
    
    config = Configuration(
        server_api_url=os.environ["CONDUCTOR_SERVER_URL"],
        authentication_settings=AuthenticationSettings(
            key_id=os.environ["CONDUCTOR_AUTH_KEY"],
            key_secret=os.environ["CONDUCTOR_AUTH_SECRET"]
        )
    )

    if args.mock_eval:
        print("Peforming a mocked run of agent behavior.")
        mock_result = mock_run(
            support_agent,
            _PROMPT,
            events=[
                MockEvent.handoff("billing"),
                MockEvent.tool_call("lookup_order", {"order_id": "123"}),
                MockEvent.tool_result("lookup_order", result={"order_id": "123", "status": "shipped"}),
                MockEvent.tool_call("process_refund", args={"order_id": "123", "amount": 49.99}),
                MockEvent.tool_result("process_refund", result="Refund of $49.99 processed"),
                MockEvent.done("Your refund request for order #123 is has been processed."),
            ],
            auto_execute_tools=True,
        )
        mock_result.print_result()
        assert_handoff_to(mock_result, "billing")
        assert_tool_call_order(mock_result, ["lookup_order", "process_refund"])
        assert_tool_not_used(mock_result, "search_web")
        assert_output_contains(mock_result, "refund", case_sensitive=False)
    else:
        with AgentRuntime(configuration=config) as runtime:
            # Always deploy updated agents when using a real runtime
            runtime.deploy(support_agent, billing_agent, technical_agent)

            print("Performing agentic sub-workflow runs.")
            result = runtime.run(agent=support_agent, prompt=_PROMPT)
            result.print_result()


if __name__=="__main__":
    main()