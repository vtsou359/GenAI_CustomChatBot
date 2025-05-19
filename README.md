# Fashion Trends RAG Chatbot

A Retrieval Augmented Generation (RAG) chatbot specialized in 2023 fashion trends. This project demonstrates how to build a custom chatbot that leverages both OpenAI and Ollama models to provide accurate, context-rich responses about fashion trends.

## Project Overview

This project implements a RAG-style chatbot that:

1. Uses a dataset of 2023 fashion trends extracted from various fashion websites
2. Processes and embeds the data to enable semantic search
3. Retrieves relevant context based on user queries
4. Generates accurate, source-backed responses using either OpenAI or Ollama models
5. Provides a web interface for easy interaction

The chatbot demonstrates the power of combining retrieval-based and generative approaches to create more accurate and informative responses compared to using large language models alone.

## Project Structure

```
GenAI_CustomChatBot/
├── data/                      # Raw and processed data files
│   └── 2023_fashion_trends.csv
├── data_chatbot/              # Processed data with embeddings for the chatbot
│   └── 2023_fashion_trends_embeddings.csv
├── fncs/                      # Core functionality modules
│   ├── chatb.py               # Main chatbot wrapper functions
│   ├── prompt_templates.py    # Prompt templates for the chatbot
│   ├── retrieval.py           # Functions for embedding and retrieval
│   └── utilities.py           # Utility functions for API clients, etc.
├── ntbk1_DataWrangling.ipynb       # Data processing and embedding generation
├── ntbk2_CustomChatBot.ipynb       # OpenAI-based chatbot implementation
├── ntbk3_Ollama_CustomChatBot.ipynb # Experimental Ollama-based implementation
├── .env                       # Environment variables (API keys, etc.)
├── chatbot.py                 # Chainlit web interface implementation
└── requirements.txt           # Project dependencies
```

## Notebooks

### 1. ntbk1_DataWrangling.ipynb

This notebook handles the data preparation process:
- Loads the 2023 fashion trends dataset
- Processes the data to create text chunks with brand, source, trends, and URL information
- Generates embeddings using either OpenAI or Ollama models
- Saves the processed data with embeddings for later use in the chatbot

### 2. ntbk2_CustomChatBot.ipynb

This notebook implements and demonstrates the OpenAI-based chatbot:
- Loads the processed data with embeddings
- Implements the RAG approach with semantic search and context retrieval
- Controls token usage to stay within model limits
- Demonstrates the chatbot's performance with example queries
- Compares responses with and without context to show the benefits of the RAG approach
- Calculates and displays token usage and costs

### 3. ntbk3_Ollama_CustomChatBot.ipynb (Experimental)

This notebook is an experimental implementation that uses Ollama instead of OpenAI:
- Similar structure to ntbk2 but uses local Ollama models
- Currently uses Gemma 3:1b for chat and granite-embedding for embeddings
- Uses a temporary tokenizer solution (OpenAI's tokenizer)
- This notebook is still under development and not finalized

## Usage

### Prerequisites

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Set up your environment variables in a `.env` file:
   ```
   OPENAI_API=your_openai_api_key
   OPENAI_BASE=https://api.openai.com/v1
   ```

3. For Ollama functionality (optional):
   - Install Ollama from [https://ollama.ai/](https://ollama.ai/)
   - Pull the required models:
     ```
     ollama pull gemma3:1b
     ollama pull granite-embedding
     ```

### Running the Notebooks

The notebooks can be run in sequence to understand the full pipeline:
1. Run `ntbk1_DataWrangling.ipynb` to process the data and generate embeddings
2. Run `ntbk2_CustomChatBot.ipynb` to see the OpenAI-based chatbot in action
3. (Optional) Run `ntbk3_Ollama_CustomChatBot.ipynb` to experiment with the Ollama-based version

### Running the Web Interface

To start the Chainlit web interface:

```
chainlit run chatbot.py
```

This will start a local web server where you can interact with the chatbot, adjust settings, and see usage statistics.

## Implementation Details

### RAG Approach

The chatbot uses a Retrieval Augmented Generation (RAG) approach:
1. **Embedding**: User queries are embedded using the same model used for the dataset
2. **Retrieval**: Relevant context is retrieved based on cosine similarity between the query and dataset embeddings
3. **Generation**: The retrieved context is combined with the query in a structured prompt
4. **Response**: The LLM generates a response based on the provided context

### Token Management

The system includes token management to control context size:
- Calculates token usage for system and user prompts
- Dynamically adjusts the amount of context to stay within token limits
- Prioritizes the most relevant context based on similarity scores

## Note on Experimental Features

The Ollama-based implementation (`ntbk3_Ollama_CustomChatBot.ipynb`) is experimental and still under development. Known limitations:
- Uses OpenAI's tokenizer instead of a proper Ollama tokenizer
- May have performance and accuracy differences compared to the OpenAI version
- Requires local installation of Ollama and specific models


## Acknowledgments

- The fashion trends data was extracted from various fashion websites
- The project uses OpenAI and Ollama for embeddings and text generation
- Chainlit is used for the web interface
