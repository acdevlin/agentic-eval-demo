from conductor.ai.agents import Agent, AgentRuntime, Strategy
from conductor.ai.agents.testing import CorrectnessEval, EvalCase
from conductor.client.configuration.configuration import Configuration
from conductor.client.configuration.settings.authentication_settings import AuthenticationSettings
import os

from agents import support_agent, billing_agent, technical_agent


def main():
    """
    The runtime is NOT mocked - all LLM calls are actually evaluated.
    """
    config = Configuration(
        server_api_url=os.environ["CONDUCTOR_SERVER_URL"],
        authentication_settings=AuthenticationSettings(
            key_id=os.environ["CONDUCTOR_AUTH_KEY"],
            key_secret=os.environ["CONDUCTOR_AUTH_SECRET"]
        )
    )

    with AgentRuntime(configuration=config) as runtime:
        # Ensure that all 3 agents are deployed and can run before we evaluate them
        runtime.deploy(support_agent, billing_agent, technical_agent)
        result = runtime.run(agent=support_agent, prompt="I need a refund for order 123.")
        result.print_result()


if __name__=="__main__":
    main()