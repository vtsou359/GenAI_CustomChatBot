import pandas as pd
import os
from pathlib import Path
from dotenv import load_dotenv
import tiktoken
import ast
import chainlit as cl

# Custom Functions
from fncs.utilities import (
    create_openai_client,
    response_generator,
    prompt_builder,
    calculate_total_cost
)
from fncs.retrieval import (
    get_embedding,
    search_text,
    control_chunk_context
)
from fncs.prompt_templates import user_prompt



def process_query(csv_path, query, max_token_count=1000):
    """
    Process a user query using the fashion trends dataset.

    Args:
        csv_path (str): Path to the CSV file with embeddings
        query (str): User query text
        max_token_count (int): Maximum token count for context (default: 1000)

    Returns:
        dict: Dictionary containing response text, cost, and usage statistics
    """
    # Load environment vars
    load_dotenv()
    base_url_voc = os.getenv("OPENAI_BASE_VOC")
    api_key_voc = os.getenv("OPENAI_API_VOC")

    # Deployment model names
    chat_name = 'gpt-4o'
    emb_name = 'text-embedding-3-large'

    # Initialize OpenAI client and tokenizer
    openai_client = create_openai_client(api_key=api_key_voc, base_url=base_url_voc)
    tokenizer = tiktoken.encoding_for_model(chat_name)

    # Load and prepare the dataset
    df = pd.read_csv(csv_path)
    df['embeddings'] = df['embeddings'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # Generate query embedding
    query_emb = get_embedding(text=query, client=openai_client, model=emb_name)

    # Sort dataframe based on cosine distance
    df_sorted = search_text(df=df, embs_query=query_emb, cosine='distance')

    # Create system prompt
    system_prompt = ("You are an expert fashion trend analyser. "
                     "Based only on the provided information "
                     "you must analyse and summarise the trends and provide an accurate answer.")


    # Calculate token counts and create context
    current_token_count = len(tokenizer.encode(user_prompt())) + len(tokenizer.encode(system_prompt))
    context = control_chunk_context(
        chunks_sorted_df=df_sorted,
        current_token_count=current_token_count,
        max_token_count=max_token_count,
        tokenizer=tokenizer
    )

    # Format the prompt with context
    context_inprompt = "\n----\n".join(context)
    user_prompt_formatted = user_prompt().format(query, context_inprompt)

    # Build the final prompt and generate response
    final_prompt = prompt_builder(system_content=system_prompt, user_content_prompt=user_prompt_formatted)
    additional_options = {"temperature": 0.4}

    response, response_full = response_generator(
        openai_client,
        chat_model=chat_name,
        prompts=final_prompt,
        options=additional_options
    )

    # Calculate cost
    cost_eur = calculate_total_cost(response_usage=response_full.usage, deployment_name=chat_name)

    # Return results
    return {
        "response": response,
        "cost": cost_eur,
        "total_tokens": response_full.usage.total_tokens,
        "completion_tokens": response_full.usage.completion_tokens,
        "prompt_tokens": response_full.usage.prompt_tokens
    }