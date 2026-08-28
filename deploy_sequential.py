#!/usr/bin/env python3
from conductor.ai.agents import AgentRuntime
from conductor.client.configuration.configuration import Configuration
from conductor.client.configuration.settings.authentication_settings import (
    AuthenticationSettings,
)
import os

from agents import content_pipeline


def main():
    """
    A sequential 3-agent pipeline used to research, summarize, and publish a simple article on a
    subject provided by the user.
    """

    config = Configuration(
        server_api_url=os.environ["CONDUCTOR_SERVER_URL"],
        authentication_settings=AuthenticationSettings(
            key_id=os.environ["CONDUCTOR_AUTH_KEY"],
            key_secret=os.environ["CONDUCTOR_AUTH_SECRET"],
        ),
    )

    with AgentRuntime(configuration=config) as runtime:
        #
        runtime.deploy(content_pipeline)
        result = runtime.run(
            content_pipeline,
            prompt=(
                "Search the web for articles about the history of cheesemaking, then "
                "write me a brief summary of under 500 words explaining the key points."
            ),
        )
        result.print_result()
        print(
            "Full run in the Orkes Conductor UI: "
            f"https://developer.orkescloud.com/agentExecutions/{result.execution_id}"
        )


if __name__ == "__main__":
    main()
