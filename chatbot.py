import pandas as pd
import os
from pathlib import Path
from dotenv import load_dotenv
import tiktoken
import ast
import chainlit as cl


# custom function process wrapper
from fncs.chatb import process_query


# Chainlit implementation
@cl.on_chat_start
def on_chat_start():
    # Set the CSV file path as a session variable
    csv_path = str(Path(os.getcwd()) / "data" / "2023_fashion_trends_embeddings.csv")
    cl.user_session.set("csv_path", csv_path)


@cl.on_message
async def main(message: cl.Message):
    """
    This function is called every time a user inputs a message in the UI.
    It processes the query and returns a fashion trend analysis.

    Args:
        message: The user's message.

    Returns:
        None.
    """
    # Get the CSV path from the session
    csv_path = cl.user_session.get("csv_path")
    query = message.content

    # Show processing message
    processing_msg = await cl.Message(content="Processing your query...", author= "Fashion Advisor").send()

    # Create a step to show the embedding and retrieval process
    step = cl.Step(name="Retrieving relevant fashion information", type="tool")
    await step.send()

    # First step message
    await cl.Message(
        content="Finding relevant fashion information for your query...",
        parent_id=step.id
    ).send()

    # Process the query and get results (this might take some time)
    results = process_query(csv_path=csv_path, query=query)

    # Update with completion message
    await cl.Message(
        content="Found relevant fashion information! Processing with AI model...",
        parent_id=step.id
    ).send()

    # Complete the step
    #await step.complete()

    # Send the final response
    response_content = f"{results['response']}\n\n---\n*Query processing cost: {results['cost']} EUR*"
    await cl.Message(content=response_content).send()

    # Optional: Show token usage in a separate message
    token_info = f"Token usage: {results['total_tokens']} total tokens ({results['prompt_tokens']} prompt, {results['completion_tokens']} completion)"
    await cl.Message(content=token_info, author="System").send()
