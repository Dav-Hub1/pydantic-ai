import asyncio
import os
from typing import List, Optional
from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext
from pydantic import BaseModel, Field

# Load environment variables from .env file
load_dotenv()

model_name = os.getenv("TRIAGE_MODEL")  # Default to a specific model if not set in .env

class CustomerSupportAgentOutput(BaseModel):
    response_message: str = Field(..., description="Message to be sent to the customer")
    escalation_needed: bool = Field(..., description="Indicates if escalation to a human support agent is needed")
    follow_up_required: bool = Field(..., description="Indicates if a follow-up is required with the customer")
    sentiment_analysis: str = Field(..., description="Sentiment analysis of the customer's message (e.g., positive, neutral, negative)")

class Order(BaseModel):
    "Order details for customer support inquiries"
    order_id: str
    status: str
    items: List[str]

# Customer details for support inquiries, including order information
class CustomerDetails(BaseModel):
    "Customer details for support inquiries"
    customer_id: str
    name: str
    email: str
    orders: Optional[List[Order]] = None

# Mock shipping database for demonstration purposes
Shipping_DB = {
    "A123": {"status": "Shipped", "estimated_delivery": "2024-06-10"},
    "B456": {"status": "Processing", "estimated_delivery": "2024-06-15"},
}

customer_support_agent = Agent[CustomerDetails, CustomerSupportAgentOutput](
    model=model_name,
    output_type=CustomerSupportAgentOutput,
    system_prompt=(
        """You are a customer support assistant.
        Your role is to provide helpful and accurate responses to customer inquiries.
        Always be polite, professional, and concise in your responses.
        
        Answer with less than 71 tokens, and ensure that the response is clear and actionable."""
    ),
    deps_type=CustomerDetails,
    retries=3,
)

# ============ SYSTEM PROMPT ============

# Add dynamic system prompt to enrich customer details with order information before generating a response
@customer_support_agent.system_prompt
async def enrich_customer_details(ctx: RunContext[CustomerDetails]) -> str:
    customer_details = ctx.deps
    if customer_details.orders is None:
        # Simulate fetching order details from a database or API
        await asyncio.sleep(0.1)  # Simulating I/O delay
        customer_details.orders = [
            Order(order_id="A123", status="Shipped", items=["Widget A", "Widget B"]),
            Order(order_id="B456", status="Processing", items=["Widget C"]),
        ]
    return f"Customer Details: {customer_details.model_dump_json()}"

# ============ TOOLS ============

@customer_support_agent.tool()
async def get_shipping_status(
    ctx: RunContext[CustomerDetails],
    order_id: str
) -> str:
    """Get shipping status for a specific order by its ID."""
    shipping_info = Shipping_DB.get(order_id)

    if shipping_info is None:
        return f"No shipping information found for order {order_id}"

    return (
        f"Order {order_id}: "
        f"Status: {shipping_info['status']}, "
        f"Estimated Delivery: {shipping_info['estimated_delivery']}"
    )

@customer_support_agent.tool()
async def get_all_orders_status(
    ctx: RunContext[CustomerDetails]
) -> str:
    """Get shipping status for all customer orders."""
    customer_details = ctx.deps

    if not customer_details.orders:
        return "No orders found for this customer."

    results = []
    for order in customer_details.orders:
        shipping_info = Shipping_DB.get(order.order_id)
        if shipping_info:
            results.append(
                f"Order {order.order_id}: "
                f"Status: {shipping_info['status']}, "
                f"Estimated Delivery: {shipping_info['estimated_delivery']}"
            )
        else:
            results.append(
                f"Order {order.order_id}: No shipping info available"
            )

    return "\n".join(results)

customer = CustomerDetails(
    customer_id="C789",
    name="Alice Johnson",
    email="alice.johnson@example.com",
    orders= [Order(order_id="B456", status="Processing", items=["Widget C"])]
    )

response = customer_support_agent.run_sync(
    user_prompt="When is my order expected to be delivered?", deps=customer)

print(response)