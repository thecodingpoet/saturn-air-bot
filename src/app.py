import gradio as gr

from query import answer_query


def chatbot_interface(message: str, history: list) -> str:
    """
    Process user message and return Saturn Airlines assistant response.

    Args:
        message: User's question
        history: Chat history (handled by Gradio)

    Returns:
        Assistant's response
    """
    result = answer_query(message)
    return result["system_answer"]


def main():
    """Launch the Gradio chat interface."""
    ui = gr.ChatInterface(
        chatbot_interface,
        examples=[
            "What are your flight policies?",
            "How do I book a ticket?",
            "What is your baggage policy?",
        ],
        title="🪐 Saturn Airlines Customer Service Assistant",
        description="Ask any questions about Saturn Airlines services, policies, flights, and bookings.",
    )

    ui.launch(inbrowser=True)


if __name__ == "__main__":
    main()
