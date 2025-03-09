"""Retrieval functions for embeddings and search."""

from typing import Any, List
import numpy as np
from scipy import spatial
import pandas as pd


def create_embeddings_batch(client,
                            deployment_name: str,
                            batch_size: int,
                            df: pd.DataFrame,
                            chunk_column: str,
                            cost_per_thousand_tokens: float = 0.000125
                            ) -> tuple[pd.DataFrame, Any]:
    """
    Creates embeddings for a given dataframe in batches by interacting with the client API. This function computes
    embeddings for the text data in a specified column of the dataframe, adds the generated embeddings as a new
    column, and calculates the total cost based on the number of tokens processed.

    :param client: The client object used for interacting with the external embeddings API.
    :param deployment_name: The name of the deployed embedding model to use for generating embeddings.
    :param batch_size: The number of rows from the dataframe to process in each batch.
    :param df: A pandas DataFrame containing the input data for which embeddings are to be generated.
    :param chunk_column: The name of the column in the dataframe that contains the text data for embeddings.
    :param cost_per_thousand_tokens: The cost per 1,000 tokens, used to calculate the total cost of processing.
        Defaults to 0.000125 (price for: text-embedding-3-large) if not specified.
    :return: A tuple containing the input dataframe with an additional "embeddings" column and the total cost
        incurred for processing the embeddings.
    :rtype: tuple(pd.DataFrame, float)
    :raises ValueError: If the input dataframe is not a pandas DataFrame, the specified column is missing, or
        if batch_size is less than 1.
    :raises RuntimeError: If an error occurs during the embedding creation process.
    """
    # Input validation
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input df must be a pandas DataFrame")

    if chunk_column not in df.columns:
        raise ValueError(f"Column {chunk_column} not found in dataframe")

    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    # Initialize empty list for embeddings
    embeddings = []
    total_tokens = 0
    try:
        # Process in batches
        for i in range(0, len(df), batch_size):
            # Get batch of texts
            batch_texts = df.iloc[i:i+batch_size][chunk_column].tolist()

            # Generate embeddings for each row in the batch
            for text in batch_texts:
                emb_response = client.embeddings.create(
                    model= deployment_name,
                    input= text.replace("\n", " ")
                )
                # Extract and append the embedding
                embedding = emb_response.data[0].embedding
                embeddings.append(embedding)

                # calculate total tokens
                total_tokens += emb_response.usage.prompt_tokens

        # Add embeddings to the dataframe
        df["embeddings"] = embeddings

        # Calculate total cost
        cost = (total_tokens / 1_000) * cost_per_thousand_tokens

        return df, cost

    except Exception as e:
        raise RuntimeError(f"Failed to create embeddings: {str(e)}")


def get_embedding(text: str, client: Any, model="text-embedding-3-small", **kwargs) -> List[float]:
    """
    Computes the embedding vector for a given text using the specified client and model. The function
    replaces newline characters in the input text to mitigate potential negative effects on
    performance and then uses the provided client to generate the embedding.

    :param text: The input text to be embedded.
    :param client: The client instance used for generating the embedding.
    :param model: The model identifier specifying the embedding model to be used. Defaults to
        "text-embedding-3-large".
    :param kwargs: Additional parameters to be passed to the client's embedding creation method.
    :return: The embedding vector of the input text as a list of floats.
    :rtype: List[float]
    """
    # replace newlines, which can negatively affect performance.
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model, **kwargs)
    return response.data[0].embedding


def cosine_distance(a, b):
    """
    Computes the cosine distance between two vectors.

    Parameters:
        a: First vector
        b: Second vector

    Returns:
        Cosine distance between input vectors, as a float.
    """
    # Convert inputs to numpy arrays and ensure they're floating point numbers
    #a = np.asarray(a, dtype=np.float32)
    #b = np.asarray(b, dtype=np.float32)

    return 1 - (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def distances_from_embeddings(
    query_embedding: List[float],
    embeddings: List[List[float]],
    distance_metric="cosine",
) -> List[List]:

    """Return the distances between a query embedding and a list of embeddings."""

    distance_metrics = {
        "cosine": spatial.distance.cosine,
        "L1": spatial.distance.cityblock,
        "L2": spatial.distance.euclidean,
        "Linf": spatial.distance.chebyshev,
    }
    distances = [
        distance_metrics[distance_metric](query_embedding, embedding)
        for embedding in embeddings
    ]
    return distances


# search through the reviews for a specific product
def search_text(df, embs_query, n=0, cosine='distance'):
    """
    Search for similar texts using embeddings.

    Parameters:
        df: DataFrame containing the embeddings
        embs_query: Query embedding
        n: Number of results to return (0 for all)
        cosine: Type of cosine calculation ('distance' or 'similarity')

    Returns:
        DataFrame with results sorted by similarity
    """
    if cosine == 'similarity':
        df["similarity"] = df.embeddings.apply(
            lambda x: 1 - cosine_distance(x, embs_query)
        )
        if n == 0:
            results = df.sort_values("similarity", ascending=False)
        else:
            results = df.sort_values("similarity", ascending=False).head(n)
        return results

    elif cosine == 'distance':
        # Ensure embeddings are properly converted to numerical arrays
        df["distance"] = df.embeddings.apply(
            lambda x: cosine_distance(x, embs_query)
        )
        if n == 0:
            results = df.sort_values("distance", ascending=True)
        else:
            results = df.sort_values("distance", ascending=True).head(n)
        return results



# Function to add chunks to the context
def control_chunk_context(chunks_sorted_df,
                          current_token_count,
                          max_token_count,
                          tokenizer
                          ):
    """
    Adds chunks of text to the context while respecting the maximum token count limit.

    This function iterates through a DataFrame containing sorted text chunks, calculates
    the token count for each text chunk, and appends them to a context list as long as
    the total token count does not exceed the specified maximum. It stops adding chunks
    once the token limit is reached.

    :param chunks_sorted_df: The DataFrame containing text chunks sorted in a specific order.
    :type chunks_sorted_df: pandas.DataFrame
    :param current_token_count: The current token count before adding any text chunk.
    :type current_token_count: int
    :param max_token_count: The maximum allowable token count for the context.
    :type max_token_count: int
    :param context: A list that will contain the text chunks added within the token limit.
    :type context: list
    :param tokenizer: The tokenizer function used to encode text into tokens. Default is
        the tokenizer for the "gpt-4o" model.
    :type tokenizer: Callable
    :return: A tuple containing the updated token count and the updated context list.
    :rtype: tuple
    """
    # Initialize the context list
    context = []
    for text in chunks_sorted_df["text"].values:
        # Calculate the token count for the current text
        text_token_count = len(tokenizer.encode(text))
        # Check if adding this text exceeds the token limit
        if current_token_count + text_token_count <= max_token_count:
            # If within limit, add text to context and update token count
            context.append(text)
            current_token_count += text_token_count
        else:
            # If token limit is exceeded, stop adding chunks
            break

    return context