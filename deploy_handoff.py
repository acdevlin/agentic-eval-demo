#!/usr/bin/env python3
""""""

from conductor.ai.agents import AgentRuntime
from conductor.client.configuration.configuration import Configuration
from conductor.client.configuration.settings.authentication_settings import (
    AuthenticationSettings,
)
import os

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

    config = Configuration(
        server_api_url=os.environ["CONDUCTOR_SERVER_URL"],
        authentication_settings=AuthenticationSettings(
            key_id=os.environ["CONDUCTOR_AUTH_KEY"],
            key_secret=os.environ["CONDUCTOR_AUTH_SECRET"],
        ),
    )

    with AgentRuntime(configuration=config) as runtime:
        # Always deploy updated agents when using a real runtime
        runtime.deploy(support_agent, billing_agent, technical_agent)
        print("Performing live agent run.")
        result = runtime.run(agent=support_agent, prompt=_PROMPT_REFUND)
        result.print_result()
        print(
            "Full run in the Orkes Conductor UI: "
            f"https://developer.orkescloud.com/agentExecutions/{result.execution_id}"
        )


if __name__ == "__main__":
    main()
