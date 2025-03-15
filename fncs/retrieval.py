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
    Creates a batch of embeddings from the provided text data in the dataframe and computes the
    total cost for the embedding operation.

    This function processes text data within a specified chunk column of the dataframe, generates
    embeddings in batches, appends the embeddings to the dataframe, and calculates the cost
    incurred based on the number of tokens processed and the cost per thousand tokens.

    :param client: The client object used for generating embeddings.
    :param deployment_name: The name of the deployed model to create embeddings.
    :param batch_size: The number of rows to process in each batch.
    :param df: The pandas DataFrame containing the data with text for embedding.
    :param chunk_column: The column in the dataframe containing the text data to embedd.
    :param cost_per_thousand_tokens: The cost per thousand tokens used in calculating the total cost.
        Defaults to 0.000125.
    :return: A tuple consisting of the updated dataframe with embeddings and the computed total cost.
    :raises ValueError: If the dataframe (`df`) is not a pandas DataFrame, the `chunk_column` is not
        present in the dataframe, or `batch_size` is not positive.
    :raises RuntimeError: If an exception occurs during the embedding creation process.
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
            batch_texts = df.iloc[i:i + batch_size][chunk_column].tolist()

            # Generate embeddings for each row in the batch
            for text in batch_texts:
                # Skip empty or None texts
                if not text or not isinstance(text, str):
                    embeddings.append([])  # Add empty embedding for invalid text
                    continue

                emb_response = client.embeddings.create(
                    model=deployment_name,
                    input=text.replace("\n", " ")
                )

                # Check if response and its components are valid
                if (emb_response and hasattr(emb_response, 'data') and
                        emb_response.data and len(emb_response.data) > 0 and
                        hasattr(emb_response.data[0], 'embedding')):
                    # Extract and append the embedding
                    embedding = emb_response.data[0].embedding
                    embeddings.append(embedding)

                    # Calculate total tokens if usage information is available
                    if hasattr(emb_response, 'usage') and hasattr(emb_response.usage, 'prompt_tokens'):
                        total_tokens += emb_response.usage.prompt_tokens
                else:
                    # Handle invalid response by adding an empty embedding
                    print(f"Warning: Invalid embedding response for text: {text[:50]}...")
                    embeddings.append([])

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