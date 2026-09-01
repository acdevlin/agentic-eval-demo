import pytest
from agents import content_pipeline
from conductor.ai.agents.testing import (
    MockEvent,
    StrategyViolation,
    mock_run,
    assert_agent_ran,
    assert_tool_used,
    assert_tool_called_with,
    validate_strategy,
)


class TestSequentialPipeline:

    def test_agent_order(self):
        """Verifies that the agents run in the specified order."""
        result = mock_run(
            content_pipeline,
            "Search for articles about AI safety, then write a summary about them.",
            events=[
                MockEvent.handoff("search"),
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
        assert_agent_ran(result, "search")
        assert_agent_ran(result, "summary")
        assert_agent_ran(result, "editor")
        # Confirm that specified tools were used
        assert_tool_used(result, "search_web")
        assert_tool_called_with(result, "search_web", args={"query": "AI safety"})

    def test_skipped_agent_throws_error(self):
        """Confirms that if an agent is skipped, an error is thrown."""
        result = mock_run(
            content_pipeline,
            "Write about agentic AI",
            events=[
                MockEvent.handoff("search"),
                # Handoff to "summary" agent is intentionally missing!
                MockEvent.handoff("editor"),
                MockEvent.done("Incomplete article"),
            ],
        )

        result.print_result()

        # These two agents DID run
        assert_agent_ran(result, "search")
        assert_agent_ran(result, "editor")

        # This agent DID NOT run
        with pytest.raises(AssertionError, match="summary"):
            assert_agent_ran(result, "summary")

        # Verify that the "summary" agent was skipped and caused a Strategy error
        with pytest.raises(StrategyViolation, match="skipped"):
            validate_strategy(content_pipeline, result)
