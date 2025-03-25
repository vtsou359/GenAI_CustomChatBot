"""Utility functions."""

from typing import Any, Callable, Dict, List, Optional, Tuple
from openai import AzureOpenAI, OpenAI

def create_openai_client(
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        organization: Optional[str] = None
) -> OpenAI:
    """
    Create an OpenAI client instance.

    This function initializes and returns an instance of the OpenAI client,
    configured with the specified API key, base URL, and optionally an
    organization. The API key is mandatory, whereas providing a base URL is
    optional and defaults to "https://api.openai.com/v1" if not supplied.
    An organization can also be included to configure the client for a
    specific organization.

    :param api_key: The API key used for authenticating with the OpenAI API.
    :type api_key: str
    :param base_url: The base URL of the OpenAI API, with a default value of
        "https://api.openai.com/v1".
    :type base_url: str, optional
    :param organization: Optional identifier for an organization to associate
        with the client.
    :type organization: Optional[str]
    :return: An instance of the OpenAI client.
    :rtype: OpenAI
    :raises ValueError: If the API key is not provided.
    """
    if not api_key:
        raise ValueError("An API key must be provided.")

    if organization:
        return OpenAI(
            api_key=api_key,
            base_url=base_url,
            organization=organization
        )
    else:
        return OpenAI(api_key=api_key, base_url=base_url)




def create_ollama_client(
        api_key: str = 'ollama',
        base_url: str = 'http://localhost:11434/v1/',
) -> OpenAI:
    """
    Creates and returns an instance of the OpenAI client initialized with the provided API key
    and base URL. This function ensures that a valid API key is supplied for authentication.

    :param api_key: The API key required for authenticating with the OpenAI API.
    :param base_url: The base URL of the OpenAI API endpoint.
    :return: An instance of the OpenAI client initialized with the given credentials and
             configuration.

    :raises ValueError: If the API key is not provided.
    """
    if not api_key:
        raise ValueError("An API key must be provided.")

    client = OpenAI(
        base_url=base_url,
        # required but ignored
        api_key=api_key,
    )
    return client



def create_azure_client(
        api_version: str,
        azure_endpoint: str,
        ad_token: Optional[str] = None,
        api_key: Optional[str] = None,
        ad_token_provider: Optional[Callable[[], Optional[str]]] = None
) -> AzureOpenAI:
    """
    Creates and returns an instance of AzureOpenAI client.

    This function is intended to initialize
    a client object to interact with the Azure OpenAI service. The client may be authenticated
    using either an Azure AD token, a dynamic token provider, or an API key. If none of these
    authentication methods is provided, an exception is raised. The `api_version` and
    `azure_endpoint` parameters are required for configuring the client.

    :param api_version: The API version of the Azure OpenAI service to be used.
    :param azure_endpoint: The endpoint of the Azure OpenAI service.
    :param ad_token: Azure AD token for authentication, provided as a string. Defaults to None.
    :param api_key: API key for authentication, provided as a string. Defaults to None.
    :param ad_token_provider: A callable function that provides an Azure AD token dynamically.
        When called, it should return the token as a string or None. Defaults to None.
    :return: An instance of AzureOpenAI configured with the provided parameters.
    :raises ValueError: If neither `ad_token`, `api_key`, nor `ad_token_provider` is provided.
    :raises ValueError: If more than one of `ad_token`, `api_key`, or `ad_token_provider` is provided.
    """
    if sum(bool(x) for x in [ad_token, api_key, ad_token_provider]) > 1:
        raise ValueError("Provide only one of `api_key`, `ad_token`, or `ad_token_provider`, not multiple.")

    if ad_token:
        return AzureOpenAI(azure_ad_token=ad_token, api_version=api_version, azure_endpoint=azure_endpoint)
    elif api_key:
        return AzureOpenAI(api_key=api_key, api_version=api_version, azure_endpoint=azure_endpoint)
    elif ad_token_provider:
        token = ad_token_provider()
        if not token:
            raise ValueError("The token provider did not return a valid token.")
        return AzureOpenAI(azure_ad_token=token, api_version=api_version, azure_endpoint=azure_endpoint)
    else:
        raise ValueError("Either `api_key`, `ad_token`, or `ad_token_provider` must be provided.")


def response_generator(
        client: Any,
        chat_model: str,
        prompts: List[Dict[str, Any]],
        options: Optional[Dict[str, Any]] = None
) -> Tuple[str, Any]:
    """

    Azure OpenAI Response Generator function with customizable options.

    :param client: The Azure OpenAI/ OpenAI client instance.
    :param chat_model: The specific chat model to use for generating responses.
    :param prompts: The list of prompt messages to be sent to the chat model.
    :param options: A dictionary of additional options for the OpenAI API call (e.g., temperature, max_tokens).
                    Example: {"temperature": 0.2, "max_tokens": 500}
    :return: A tuple containing the content of the first choice's message and the entire response object.

    """
    if options is None:
        # Default options if no custom options are provided
        options = {}

    # Prepare parameters for the API call, combining the mandatory ones with additional options from the user
    parameters = {"model": chat_model, "messages": prompts}
    parameters.update(options)  # Unpack `options` dictionary and merge with default parameters

    # Call Azure OpenAI chat completion endpoint with dynamic parameters
    response = client.chat.completions.create(**parameters)

    return response.choices[0].message.content, response


def prompt_builder(
        system_content: str,
        user_content_prompt: str
) -> List[Dict[str, str]]:
    """

    Prompt Builder function.

    :param system_content: The content or message that defines the system's prompt in the conversation.
    :param user_content_prompt: The content or message provided by the user in the conversation.
    :return: A list of dictionaries, each containing the role ('system' or 'user') and their respective content.
    """
    return [{"role": "system", "content": system_content}, {"role": "user", "content": user_content_prompt}]


def calculate_total_cost(
        response_usage: Any,
        deployment_name: str = "gpt-4o"
) -> float:
    """

    Total Cost Calculation function.

    :param response_usage: Data structure containing token usage details.
    :param deployment_name: The model used for token calculation (default: 'GPT-4o').
    :return: Total cost based on token usage and model type.

    """

    def calculate_cost_per_token(cost_per_million_tokens: float) -> float:
        """
        Convert cost per million tokens to cost per single token.

        :param cost_per_million_tokens: Cost for 1,000,000 tokens.
        :return: Cost per token.
        """
        return cost_per_million_tokens / 1_000_000

    # Prices found in openAI API website:
    if deployment_name == "gpt-4o-mini":
        input_cost_per_million = 0.15
        output_cost_per_million = 0.60

    if deployment_name == "gpt-4o":
        input_cost_per_million = 2.50
        output_cost_per_million = 10.00

    if deployment_name == "gpt-3.5-turbo":
        input_cost_per_million = 0.50
        output_cost_per_million = 1.50

    input_cost_per_token = calculate_cost_per_token(input_cost_per_million)
    output_cost_per_token = calculate_cost_per_token(output_cost_per_million)

    prompt_tokens = response_usage.prompt_tokens
    completion_tokens = response_usage.completion_tokens

    total_cost = (prompt_tokens * input_cost_per_token) + (completion_tokens * output_cost_per_token)
    return total_cost
