import pandas as pd
import os
from pathlib import Path
from dotenv import load_dotenv
import tiktoken
import ast
import chainlit as cl
import glob
from chainlit.input_widget import TextInput, Select

# custom function process wrapper
from fncs.chatb import process_query


# Chainlit implementation
@cl.on_chat_start
async def on_chat_start():
    # Get list of available CSV files in the data directory
    data_dir = Path(os.getcwd()) / "data_chatbot"
    csv_files = [f.name for f in data_dir.glob("*.csv")]

    # Default to the first file in the list or use the original one if no files found
    default_csv = "2023_fashion_trends_embeddings.csv"
    if not csv_files:
        csv_files = [default_csv]

    # Load environment vars for default values
    load_dotenv()
    default_api_key = os.getenv("OPENAI_API", "")

    # Set up initial values (will be overridden by user input)
    cl.user_session.set("model_name", "gpt-4o-mini")
    cl.user_session.set("api_key", default_api_key)
    cl.user_session.set("csv_path", str(data_dir / csv_files[0]))

    # Create chat settings with the appropriate widgets
    settings = await cl.ChatSettings(
        [
            TextInput(
                id="api_key",
                label="OpenAI API Key",
                initial= default_api_key,
                #password=True
            ),
            Select(
                id="model_name",
                label="OpenAI - Model",
                values=["gpt-3.5-turbo", "gpt-4o-mini"],
                initial_index=0
            ),
            Select(
                id="csv_file",
                label="CSV Dataset",
                values=csv_files,
                initial_index=0
            )
        ]
    ).send()

    # Update session variables based on settings
    cl.user_session.set("api_key", settings["api_key"])
    cl.user_session.set("model_name", settings["model_name"])

    # Set the correct CSV path based on the selected file
    csv_path = str(data_dir / settings["csv_file"])
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
    # Get the session variables
    csv_path = cl.user_session.get("csv_path")
    api_key = cl.user_session.get("api_key")
    model_name = cl.user_session.get("model_name")
    query = message.content

    # Show processing message
    await cl.Message(content="Processing your query...", author="Fashion Advisor").send()

    # Create a step to show the embedding and retrieval process
    step = cl.Step(name="Retrieving relevant fashion information", type="retrieval")
    await step.send()

    # First step message
    #await cl.Message(content="Finding relevant fashion information for your query...", parent_id=step.id).send()

    # Process the query and get results (this might take some time)
    results = process_query(csv_path=csv_path, query=query, api_key=api_key, chat_model=model_name)

    # Update with completion message
    #await cl.Message(content="Found relevant fashion information! Processing with AI model...",parent_id=step.id).send()


    # Send the final response
    response_content = \
    f"""{results['response']}
    
    Usage:
    - Query processing cost: {results['cost']} US$
    - Total Token usage: {results['total_tokens']} tokens
    - Input Tokens: {results['prompt_tokens']} tokens
    - Output Tokens {results['completion_tokens']} tokens
    """
    await cl.Message(content=response_content).send()
