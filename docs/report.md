# Technical Report

## System Architecture

### Overview

The system follows a two-phase architecture:

1. **Indexing Phase**: Documents are processed, chunked, embedded, and stored in a vector database
2. **Query Phase**: User questions are embedded, similar chunks are retrieved, and responses are generated using an LLM

### Components

#### 1. Document Processing Pipeline (`build_index.py`)

**Purpose**: Transform raw FAQ documents into searchable vector embeddings

**Process Flow**:
- **Document Loading**: Reads FAQ content from `data/faq_document.txt` using LangChain's TextLoader
- **Text Splitting**: Divides documents into manageable chunks using RecursiveCharacterTextSplitter
  - Chunk size: 500 characters
  - Overlap: 200 characters (ensures context continuity across chunks)
- **Embedding Generation**: Converts each chunk into a 3072-dimensional vector using OpenAI's `text-embedding-3-large` model
- **Vector Storage**: Persists embeddings in ChromaDB, a vector database optimized for similarity search

**Configuration Parameters**:
```python
CHUNK_SIZE = 500
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "text-embedding-3-large"
```

#### 2. Query Processing Pipeline (`query.py`)

**Purpose**: Process user queries and generate contextually grounded responses

**Process Flow**:

1. **Query Embedding**
   - User's natural language question is converted into a vector embedding
   - Uses the same embedding model (`text-embedding-3-large`) as indexing for consistency

2. **Similarity Search**
   - Performs k-nearest neighbors (KNN) search in ChromaDB
   - Retrieves top-k most semantically similar document chunks (default: k=10)
   - Uses cosine similarity for vector comparison

3. **Context Retrieval**
   - Extracts the actual text content from retrieved chunks

4. **Prompt Construction**
   - Builds a comprehensive system prompt that includes:
     - Retrieved context chunks
     - Saturn Airlines assistant guidelines
     - Out-of-scope handling instructions
   - Formats user query as a human message

5. **LLM Generation**
   - Sends prompt to `gpt-4.1-nano` language model
   - Temperature set to 0 for deterministic, factual responses
   - Model generates answer grounded in provided context

6. **Response Formatting**
   - Returns structured JSON containing:
     - Original user question
     - Generated answer
     - Retrieved chunks with metadata
     - Optional evaluation metrics

**Configuration Parameters**:
```python
MODEL = "gpt-4.1-nano"
EMBEDDING_MODEL = "text-embedding-3-large"
RETRIEVAL_K = 10
```

#### 3. Answer Evaluation System (`evaluator.py`)

**Purpose**: Assess the quality of generated responses across multiple dimensions

**Evaluation Metrics**:
- **Chunk Relevance Score**: How well retrieved chunks match the user's question
- **Answer Accuracy Score**: Correctness based on provided context
- **Completeness Score**: Whether the answer fully addresses the question
- **Tone Score**: Appropriateness of professional, friendly tone
- **Out-of-Scope Handling Score**: Quality of responses to irrelevant questions
- **Overall Score**: Weighted average of all metrics

**Output**:
- Numerical scores (0-10 scale)
- Strengths and weaknesses analysis
- Actionable improvement suggestions

### User Interfaces

#### Command-Line Interface (CLI)
- Accepts queries via `-q` argument
- Returns detailed JSON output with chunks and optional evaluation

#### Web Interface (Gradio)
- Browser-based chat interface
- Real-time conversational experience
- Simplified output (answer only, no technical details)
- Example questions for quick testing

## Technical Specifications

### Dependencies

Core libraries:
- **LangChain**: Framework for building LLM applications
- **ChromaDB**: Vector database for similarity search
- **OpenAI API**: Embeddings and language model access
- **Gradio**: Web interface framework
- **pytest**: Testing framework

### Data Flow

```
User Query
    ↓
Query Embedding (text-embedding-3-large)
    ↓
KNN Search in ChromaDB
    ↓
Retrieve Top-K Chunks
    ↓
Construct System Prompt + Context
    ↓
LLM Generation (gpt-4.1-nano)
    ↓
Return Answer + Metadata
```

## Performance Characteristics

### Embedding Model
- **Model**: text-embedding-3-large
- **Dimensions**: 3072
- **Advantages**: High-quality semantic representations, excellent accuracy
- **Cost**: ~$0.13 per 1M tokens
- **Latency**: API call required for each query

### Retrieval Strategy
- **Method**: K-nearest neighbors (KNN) with cosine similarity
- **k-value**: 10 chunks
- **Trade-off**: Higher k provides more context but may introduce noise

### Response Generation
- **Model**: gpt-4.1-nano
- **Temperature**: 0 (deterministic output)
- **Context Window**: Accommodates system prompt + 10 chunks + user query

## System Limitations

### 1. No Conversation History
- Each query is processed independently
- No memory of previous interactions within a session
- Users must provide full context in each question

**Impact**: Limits natural multi-turn conversations

**Potential Solution**: Implement conversation history

### 2. Single Document Source
- Currently supports only `data/faq_document.txt`
- No multi-document or multi-format support

**Impact**: Limited knowledge base scope

**Potential Solution**: Extend loader to support multiple files, PDFs, and other formats

### 3. No Offline Support
- Requires OpenAI API connectivity for embeddings and generation
- Incurs per-query API costs

**Impact**: Ongoing operational costs, privacy considerations, requires internet

**Potential Solution**: 
- Use local embedding models (e.g., SentenceTransformers: `all-MiniLM-L6-v2`, `all-mpnet-base-v2`)
- Deploy local LLMs (e.g., Llama, Mistral)
- Trade-offs: Lower embedding dimensions (384 vs 3072), potential accuracy reduction

### 4. Static Retrieval Strategy
- Fixed k-value regardless of query complexity
- No adaptive retrieval based on confidence scores

**Impact**: May retrieve too few or too many chunks for certain queries

**Potential Solution**: Implement dynamic k-selection based on similarity thresholds

## Proposed Improvements

### 1. Reranking
**Purpose**: Improve ordering of retrieved chunks for more relevant context

**Implementation Options**:
- Cross-encoder models (e.g., `cross-encoder/ms-marco-MiniLM-L-12-v2`)
- LLM-based reranking

**Expected Benefit**: Higher quality context leads to more accurate answers

### 2. Query Expansion
**Purpose**: Increase recall by capturing semantic variations of user queries

**Techniques**:
- Synonym expansion
- Paraphrase generation
- Multi-query retrieval

**Expected Benefit**: Better coverage of relevant information, especially for ambiguous queries

### 3. Approximate Nearest Neighbor (ANN) Indexing
**Purpose**: Faster similarity search for large-scale knowledge bases

**Implementation Options**:
- FAISS (Facebook AI Similarity Search)
- HNSW (Hierarchical Navigable Small World)

**When Needed**: Knowledge base grows beyond ~100K chunks or latency becomes critical

**Expected Benefit**: Sub-millisecond retrieval even with millions of vectors

### 4. Hybrid Search + Metadata Filters
**Purpose**: Combine keyword matching with semantic search for robust relevance

**Approach**:
- BM25 for keyword search
- Vector embeddings for semantic search
- Weighted fusion of results
- Metadata filtering (e.g., by document type, date, category)

**Expected Benefit**: Better handling of exact-match queries while maintaining semantic understanding

## Security and Privacy Considerations

### Current State
- FAQ content sent to OpenAI API for embedding and generation
- No PII detection or redaction in place

### Recommendations
1. **PII Protection**: Implement detection and redaction before API calls
2. **Data Governance**: Review what content can be sent to external APIs
4. **Audit Logging**: Track queries and responses for compliance
