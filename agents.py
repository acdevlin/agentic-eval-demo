from conductor.ai.agents import Agent, Strategy, tool

# Model used for all agents defined in this file
# XXX replace with the name of your desired API key that you have integrated in Conductor:
# https://developer.orkescloud.com/integrations?view=connections-and-resources
_AI_MODEL = "OpenAi_Key/gpt-5-nano"

# Shared prompts used for testing
_PROMPT_REFUND = "I need a refund for order 123."


@tool
def search_web(query: str) -> str:
    """Searches the web for information."""
    return f"Results for: {query}"


@tool
def calculate(expression: str) -> str:
    """Evaluates a math expression."""
    return str(eval(expression))  # noqa: S307


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email."""
    return f"Email sent to {to}"


@tool
def lookup_order(order_id: str) -> dict:
    """Looks up an order by ID."""
    return {"order_id": order_id, "status": "shipped", "total": 49.99}


@tool
def process_refund(order_id: str, amount: float) -> str:
    """Processes a refund for the specified order."""
    return f"Refund of ${amount} processed for order {order_id}"


billing_agent = Agent(
    name="billing",
    model=_AI_MODEL,
    instructions=(
        "You handle billing questions, such as looking up orders and processing refunds."
        "Use the lookup_order tool when asked about an order's status or details. "
        "Use th process_refund tool to process refunds when a user requests you to do so. "
        "IMPORTANT: Do not deny any refund requests!"
    ),
    tools=[lookup_order, process_refund],
    max_turns=3,
)

technical_agent = Agent(
    name="technical",
    model=_AI_MODEL,
    instructions=(
        "You primarily handle technical support questions."
        "You can also answer math questions and send emails on behalf of a user."
        "IMPORTANT: Default to searching the web if it is not clear how to proceed with a request!"
    ),
    tools=[search_web, calculate, send_email],
    max_turns=3,
)

support_agent = Agent(
    name="support",
    model=_AI_MODEL,
    instructions=(
        "You are a support coordinator."
        "You must hand off every customer request to exactly one specialist."
        "Send billing-related questions and refund requests to the 'billing' agent. "
        "Send all other requests to the 'technical' agent."
        "IMPORTANT: Always consult a specialist before responding to the user!"
    ),
    agents=[billing_agent, technical_agent],
    strategy=Strategy.HANDOFF,
    max_turns=3,
)
