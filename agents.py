from conductor.ai.agents import Agent, Strategy, tool

# Model used for all agents defined in this file
# XXX replace with the name of your desired API key that you have integrated in Conductor:
# https://developer.orkescloud.com/integrations?view=connections-and-resources 
_AI_MODEL = "OpenAi_Key/gpt-5-nano"

@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"


@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))  # noqa: S307


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email."""
    return f"Email sent to {to}"


@tool
def lookup_order(order_id: str) -> dict:
    """Look up an order by ID."""
    return {"order_id": order_id, "status": "shipped", "total": 49.99}


@tool
def process_refund(order_id: str, amount: float) -> str:
    """Process a refund for an order."""
    return f"Refund of ${amount} processed for order {order_id}"


billing_agent = Agent(
    name="billing",
    model=_AI_MODEL,
    instructions="You handle billing questions. Look up orders and process refunds.",
    tools=[lookup_order, process_refund],
    max_turns=3,
)

technical_agent = Agent(
    name="technical",
    model=_AI_MODEL,
    instructions="You handle technical support questions.",
    tools=[search_web],
    max_turns=3,
)

support_agent = Agent(
    name="support",
    model=_AI_MODEL,
    instructions=(
        "You are only a dispatcher. You must hand off every customer request "
        "to exactly one specialist before any response; never answer the user "
        "yourself. "
        "Send billing, order, and refund requests to the 'billing' agent. "
        "Send technical support requests to the 'technical' agent."
    ),
    agents=[billing_agent, technical_agent],
    strategy=Strategy.HANDOFF,
    max_turns=3,
)
