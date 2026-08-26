#!/usr/bin/env python3
from conductor.ai.agents import AgentRuntime
from conductor.ai.agents.testing import (
    CorrectnessEval,
    EvalCase,
)
from conductor.client.configuration.configuration import Configuration
from conductor.client.configuration.settings.authentication_settings import AuthenticationSettings
import os
import argparse

from event_capturing_runtime import EventCapturingRuntime
from agents import (
    support_agent,
    billing_agent,
    technical_agent,
    _PROMPT_REFUND,
)


def main():
    """
    Demonstrates an execution handoff between 3 agents.
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
    parser.add_argument(
        "--live-eval",
        action="store_true",
        default=False,
        help="Evaluates behavior against the live LLM."
    )
    args = parser.parse_args()

    config = Configuration(
        server_api_url=os.environ["CONDUCTOR_SERVER_URL"],
        authentication_settings=AuthenticationSettings(
            key_id=os.environ["CONDUCTOR_AUTH_KEY"],
            key_secret=os.environ["CONDUCTOR_AUTH_SECRET"]
        )
    )

    with AgentRuntime(configuration=config) as runtime:
        # Always deploy updated agents when using a real runtime
        runtime.deploy(support_agent, billing_agent, technical_agent)
        if args.live_eval:
            print("Evaluating correctness of agent behavior against live LLM.")
            billing_eval = EvalCase(
                name="refund_request_routes_to_billing",
                agent=support_agent,
                prompt=_PROMPT_REFUND,
                expect_handoff_to="billing",
                expect_tools=["lookup_order"],
                expect_tools_not_used=["search_web"],
                expect_output_contains=["refund"],
            )
            eval = CorrectnessEval(EventCapturingRuntime(runtime))
            eval_result = eval.run([
                billing_eval
            ])
            eval_result.print_summary()
            assert eval_result.all_passed, (
                f"{eval_result.fail_count}/{eval_result.total} eval(s) failed:\n"
                + "\n".join(
                    f"  - {c.name}: {[ch.message for ch in c.checks if not ch.passed]}"
                    for c in eval_result.failed_cases()
                )
            )
        else:
            print("Performing live agent run.")
            result = runtime.run(agent=support_agent, prompt=_PROMPT_REFUND)
            result.print_result()
            print("Full run in the Orkes Conductor UI: "
                    f"https://developer.orkescloud.com/agentExecutions/{result.execution_id}")


if __name__=="__main__":
    main()