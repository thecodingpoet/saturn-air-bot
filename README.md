# 🪐 Saturn Air Bot

## Description

Saturn Air Bot is an intelligent FAQ support assistant built for Saturn Airlines' customer support team. The system leverages Retrieval-Augmented Generation (RAG) to provide instant, accurate answers to passenger and staff queries based on the company's internal documentation.

## Features

- 🔍 **RAG-Powered Q&A**: Uses vector embeddings and semantic search to retrieve relevant context from FAQ documents
- 🔄 **Dual Interface**: Available via command-line interface (CLI) and web-based chat interface
- ✅ **Quality Evaluation**: Optional answer quality assessment with detailed metrics
- 🚫 **Out-of-Scope Handling**: Intelligently declines unrelated questions and provides contact information for Saturn Airlines-specific queries not in the knowledge base
- 📄 **Chunk-Based Retrieval**: Efficient document chunking with configurable size and overlap
- 📊 **Structured Output**: CLI returns detailed JSON with question, answer, retrieved chunks, and optional evaluation metrics

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for package management.

### Prerequisites

- Python 3.11 or higher
- [uv](https://github.com/astral-sh/uv) package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd saturn-air-bot
```

2. Install dependencies using uv:
```bash
uv sync
```

3. Create a `.env` file in the project root with your OpenAI API key:
```bash
OPENAI_API_KEY=your_api_key_here
```

## Building the Knowledge Base

Before using the bot, you need to build the vector index from your FAQ documents:

```bash
uv run python src/build_index.py
```

This will:
- Load documents from `data/faq_document.txt`
- Split them into chunks
- Create embeddings using OpenAI's `text-embedding-3-large` model
- Store the vector index in `chroma_db/`

## How Query Processing Works

When you submit a query, `src/query.py` performs the following steps:

- **Query Embedding**: Converts your natural language question into a vector embedding using OpenAI's `text-embedding-3-large` model
- **Similarity Search**: Performs k-nearest neighbors (KNN) search in the Chroma vector store to find the top-k most semantically similar document chunks (default: k=10)
- **Context Retrieval**: Retrieves the actual text content from the most relevant chunks
- **Prompt Construction**: Builds a system prompt that includes the retrieved context and guidelines for the AI assistant
- **LLM Generation**: Sends the prompt and user query to the language model (`gpt-4.1-nano`) to generate a grounded, contextual answer
- **Response Formatting**: Returns the answer along with metadata about retrieved chunks and optional quality evaluation

## Usage

### 1. Command-Line Interface (CLI)

The CLI version returns structured JSON output with detailed information about the query and response.

#### Basic Query
```bash
uv run python src/query.py -q "What is your baggage policy?"
```

#### With Evaluation
```bash
uv run python src/query.py -q "What is your baggage policy?" --evaluate
```

#### CLI Output Format

The CLI returns a JSON object with the following structure:

```json
{
  "user_question": "What is your baggage policy?",
  "system_answer": "Our baggage policy allows...",
  "chunks_related": [
    {
      "content": "Baggage Policy: Passengers are allowed...",
      "metadata": {
        "source": "data/faq_document.txt"
      }
    }
  ],
  "evaluation": {
    "chunk_relevance_score": 9,
    "answer_accuracy_score": 8,
    "completeness_score": 7,
    "tone_score": 9,
    "out_of_scope_handling_score": 10,
    "overall_score": 8,
    "strengths": ["..."],
    "weaknesses": ["..."],
    "improvement_suggestions": "..."
  },
  "quality_score": 8
}
```

**Note**: The `evaluation` and `quality_score` fields are only included when using the `--evaluate` flag.

### 2. Web-Based Chat Interface

For interactive chatting, use the Gradio-based web interface:

```bash
uv run python src/app.py
```

This will:
- Launch a web interface in your default browser
- Provide example questions to get started
- Allow interactive Q&A without JSON output
- Display only the assistant's response (no chunks or evaluation)

## Testing

Run the full test suite:

```bash
uv run pytest tests/
```

Run tests with verbose output:

```bash
uv run pytest tests/ -v
```

## Configuration

Key configuration parameters in `src/query.py`:

- `MODEL`: LLM model name (default: `"gpt-4.1-nano"`)
- `EMBEDDING_MODEL`: Embedding model (default: `"text-embedding-3-large"`)
- `RETRIEVAL_K`: Number of chunks to retrieve (default: `10`)

Chunking parameters in `src/build_index.py`:

- `CHUNK_SIZE`: Character size per chunk (default: `500`)
- `CHUNK_OVERLAP`: Overlap between chunks (default: `200`)

## Limitations

- **No Chat History**: The system does not maintain conversation history. Each query is processed independently without context from previous interactions.
- **Single Document Source**: Currently supports a single FAQ document (`data/faq_document.txt`)
- **No Offline Support**: Relies on OpenAI's `text-embedding-3-large` API for embeddings, requiring internet connectivity and incurring API costs. Local embedding models (e.g., `all-MiniLM-L6-v2`, `all-mpnet-base-v2`) could enable offline operation, eliminate costs, and ensure complete data privacy, though with trade-offs in embedding dimensions (384 vs 3072) and potentially slight accuracy differences.

## Improvements

- **Reranking**: More accurate ordering of retrieved chunks for better final answers.
- **Query expansion**: Higher recall by capturing paraphrases and synonyms the user didn't type.
- **ANN indexing**: When the knowledge base grows significantly (or latency becomes critical), switch to ANN to keep retrieval fast.
- **Hybrid search + metadata filters**: More robust relevance by combining exact and semantic matches while narrowing to the most pertinent chunks.