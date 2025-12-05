# RAG system
A RAG system first searches for real information from documents, websites, or databases (Retrieval).

Then the AI uses that information to create a better, more accurate answer (Generation).
# example
Imagine you are doing homework.
Instead of only using your memory, you first look it up in a book, then you write the answer.
That is exactly what RAG does.
# Simple RAG Flow
# 1. Load your data

(PDFs, text files, webpages, etc.)

# 2. Chunk the data

Break large text into small parts (chunks), so the system can search better.

# 3. Create embeddings for each chunk

Turn each chunk into a vector (a list of numbers).

# 4. Store embeddings in a vector database

Examples: Pinecone, Qdrant, Chroma, Weaviate.

# 5. At query time:

(a) Convert the user question into an embedding
(b) Search the vector database for similar chunks
(c) Retrieve the most relevant chunks

# 6. Send the retrieved chunks + the user question to the LLM

The LLM (ChatGPT, etc.) uses ONLY these chunks to give the final answer.

# CHUNKING
Chunking means cutting a big document into small pieces so the AI can search inside it better.
# Example:
If a PDF has 20 pages → chunking may break it into 40 small pieces of text (chunks).
# WHY DO WE NEED CHUNKING?
LLMs cannot take large documents directly (too many tokens).

Embeddings work best on smaller text pieces.

Searching small pieces gives more accurate results.

It reduces hallucinations because each chunk is focused.
# HOW CHUNKING WORKS
# 1. Read your document

Example: A 10,000-word PDF.

# 2. Decide your chunk size

Most common size:
Chunk size = 300–500 tokens
(1 token ≈ ¾ of a word)

Also add:
Chunk overlap = 50–100 tokens
# Overlap
Overlap  means the end of one chunk is repeated at the beginning of the next chunk.
This prevents losing meaning between chunk boundaries.
# Embedding
Embedding = Turning text into numbers so computers can search, match, and understand meaning — it’s required for RAG retrieval.
# WHY DO WE USE EMBEDDINGS?
Think of it like this:

1-You have a library of documents (PDFs, articles, etc.).

2-When someone asks a question, you could search word-by-word, but that’s slow and misses meanings (e.g., “car” and “automobile” mean the same but look different to a dumb search).

3-Embeddings convert each sentence or chunk into a list of numbers (a vector) that captures its meaning.

4-The computer then compares the numbers from your question with the numbers from all document chunks, and picks the closest matches in meaning, not just keywords.

5-Those matched chunks are fed to the LLM to write a grounded, accurate answer.
# SOURCE EMBEDDING MODELS (No API Key Required)

| Model Name | Dimensions | Best For | Notes |
| :--- | :--- | :--- | :--- |
| all-MiniLM-L6-v2 | 384 | Lightweight, fast | Default choice for many local apps |
| all-mpnet-base-v2 | 768 | High quality, balanced | Better accuracy than MiniLM |
| BAAI/bge-small-en | 384 | Good balance of speed/quality | From Beijing Academy of AI |
| BAAI/bge-base-en | 768 | Higher accuracy | Strong open-source model |
| BAAI/bge-large-en | 1024 | Best performance | Large but accurate |
| e5-large-v2 | 1024 | Instruction-aware | Can handle queries well |
| gte-small | 384 | General purpose | Good for multilingual |
| gte-base | 768 | Better than small | From Alibaba DAMO |
| multilingual-e5 | 1024 | Multiple languages | Supports 100+ languages |

# Major API-Based Embedding Models
| Provider | Model Name | Dimensions | Max Tokens | Cost (per 1K tokens) | Best For | Comparative Feature | API Key Usage |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **OpenAI** | text-embedding-3-small | 1536 | 8191 | $0.00002 | General purpose | Popular, well-documented | `Authorization: Bearer <API_KEY>` in header or initialized in Client SDK. |
| **OpenAI** | text-embedding-3-large | 3072 | 8191 | $0.00013 | High accuracy | Popular, well-documented | `Authorization: Bearer <API_KEY>` in header or initialized in Client SDK. |
| **OpenAI** | text-embedding-ada-002 | 1536 | 8191 | $0.00010 | Legacy (still good) | Popular, well-documented | `Authorization: Bearer <API_KEY>` in header or initialized in Client SDK. |
| **Cohere** | embed-english-v3.0 | 1024/384/... | 512 | $0.00010 | Enterprise use | Excellent Multilingual, Instruction-aware | `api_key="<YOUR_API_KEY>"` in SDK Client or `X-Cohere-Api-Key` header. |
| **Cohere** | embed-multilingual-v3.0 | 1024 | 512 | $0.00010 | 100+ languages | Excellent Multilingual, Instruction-aware | `api_key="<YOUR_API_KEY>"` in SDK Client or `X-Cohere-Api-Key` header. |
| **Google** | text-embedding-004 | 768 | 2048 | $0.0001/1K | Google ecosystem | Google Cloud integration | `x-goog-api-key: $GEMINI_API_KEY` in header or initialized in SDK Client. |
| **Google** | text-multilingual-embedding-002 | 768 | 2048 | $0.0001/1K | Multilingual | Google Cloud integration | `x-goog-api-key: $GEMINI_API_KEY` in header or initialized in SDK Client. |
| **Voyage AI** | voyage-2/3.5 | 1024 | 4000 (varies) | $0.0001/1K | High MTEB scores | Very Fast Speed, High accuracy | `Authorization: Bearer $VOYAGE_API_KEY` in header or initialized in SDK Client. |
| **Anthropic** | Claude embeddings* | Varies | - | Via Claude API | With Claude chats | Good Multilingual | Via Claude API (uses their main API key). |
| **Azure OpenAI** | Same as OpenAI | Same | 8191 | Similar | Enterprise Azure users | Popular, well-documented | Requires an **endpoint** and API key (`KEY1` or `KEY2`). |
| **Jina AI** | jina-embeddings-v2 | 768 | 8K | $0.014/1M tokens | Long contexts (8K) | - | `Authorization: Bearer <JINA_API_KEY>` header or `api_key` in SDK Client. |
| **Mistral AI** | mistral-embed | 1024 | - | $0.10/1M tokens | Fast, good quality | Fast Speed | `Authorization: Bearer YOUR_APIKEY_HERE` in header or `api_key` in SDK Client. |
# 2VECTOR STORE
A vector store is a special database that stores embeddings (vectors) and helps you find similar vectors very quickly.

Embeddings = numbers representing text meaning

Vector store = organized place to store and search those vectors

# WHY DO WE USE IT?
LLMs cannot search millions of chunks quickly

Vector store allows fast similarity search

Helps RAG systems find relevant chunks for user queries
# HOW VECTOR STORE WORKS
# Step 1: Store vectors

Every text chunk has an embedding vector

Save vector + metadata (doc name, chunk id, text)

# Step 2: At query time

Convert user query → embedding vector

Compare query vector with all stored vectors

Return the top-k most similar vectors

# Step 3: Feed results to LLM

LLM receives the relevant chunks

Generates answer based on these chunks
# Vector stores use vector similarity metrics:
# 1-Cosine Similarity (most common)

Measures angle between two vectors

Formula
# similarity=A.B/||A||||B||

	​
# 2- Euclidean Distance

Measures straight-line distance between two vectors in n-dimensional space
# 1. FREE VECTOR DATABASES (Open Source)
| Vector Store | Language | Features | Best For |
| :--- | :--- | :--- | :--- |
| **ChromaDB** | Python | Simple, in-memory, persistent | Beginners, prototyping |
| **FAISS (Facebook)** | C++/Python | Extremely fast similarity search | Large-scale similarity search |
| **Qdrant** | Rust/Python | Cloud-native, good performance | Production & scalability |
| **Weaviate** | Go/Python | GraphQL, hybrid search, modules | Enterprise features for free |
| **Milvus** | Go/Python | Distributed, high-performance | Large-scale production |
| **LanceDB** | Rust/Python | Disk-based, Arrow format | Multi-modal data |
| **PGvector (PostgreSQL)** | SQL | PostgreSQL extension | SQL lovers, existing PG users |
| **Redis + Redisearch** | C/Python | In-memory, real-time | Real-time applications |
| **HNSWlib** | C++/Python | Pure HNSW algorithm | Academic/research |
| **Annoy (Spotify)** | C++/Python | Approximate Nearest Neighbors | Simple ANN needs |
# A. Cloud  vector Databases/ Managed (Paid or Freemium)
| Vector Store | Pricing Model | Key Features | Best For |
| :--- | :--- | :--- | :--- |
| Pinecone | $0.10/GB-month + $0.10/1K queries | Fully managed, auto-scaling | Enterprise, no-ops |
| Weaviate Cloud | $25+/month | Managed Weaviate | Weaviate users who want cloud |
| Qdrant Cloud | $25+/month | Managed Qdrant | Qdrant users needing cloud |
| Zilliz Cloud (Milvus) | $150+/month | Managed Milvus | Large-scale enterprise |
| Vespa Cloud | $700+/month | Full-featured search engine | Complex search needs |
| Azure AI Search | $73+/month | Integrated with Azure ecosystem | Azure/Azure OpenAI users |
| Google Vertex AI | $0.10/1K queries | Google ecosystem integration | Google Cloud users |
| AWS Kendra/OpenSearch | $700+/month (Kendra) | AWS ecosystem | AWS shops |
| Marqo | $49+/month | End-to-end vector search | Simplified vector pipeline |
# Large Language Model
LLM = AI model that understands and generates human language.

Can read text, understand context, and generate answers.
# TYPES OF LLMs
LLMs are generally categorized based on architecture:

# A. Decoder-only models (Autoregressive)

Generates text one token at a time

Good for text generation, chat, summarization

# How it works:

Takes input tokens

Predicts the next token

Feeds predicted token back as input

Repeats until done

# Examples:

GPT-3, GPT-4 (OpenAI)

LLaMA series

MPT

# Use case in RAG:

Best for answer generation, chatbots, story writing
# B. Encoder-only models (Bidirectional)

Reads entire text at once to create understanding

Good for classification, embeddings, semantic search

Cannot generate text (not designed for generation)

# How it works:

Processes all tokens together using self-attention

Produces embeddings or classifications

# Examples:

BERT

RoBERTa

DistilBERT
# 1. LOCAL LLMs (100% FREE, OFFLINE)

| Family | Model | Size (Parameters) | Context (Tokens) | Best For / Specialization | Hardware / RAM Needed |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Meta (Llama)** | Llama 3.2 | 1B, 3B, 11B, 90B | 128K | Best overall | 8GB+ RAM (1B), 24GB+ (3B) |
| **Meta (Llama)** | Llama 3.1 | 8B, 70B, 405B | 128K | High quality | 16GB+ RAM (8B) |
| **Meta (Llama)** | Llama 3 | 8B, 70B | 8K | General purpose | 16GB+ RAM (8B) |
| **Meta (Llama)** | Llama 2 | 7B, 13B, 70B | 4K | Older but stable | 8GB+ RAM (7B) |
| **Microsoft (Phi)** | Phi-3.5 | 3.8B, 7B, 14B | 128K | Small but smart | 4GB+ RAM |
| **Microsoft (Phi)** | Phi-3 | 3.8B, 14B | 128K | Mobile/edge | 4GB+ RAM |
| **Microsoft (Phi)** | Phi-2 | 2.7B | 2K | Lightweight | 4GB RAM |
| **Mistral AI** | Mistral 7B v0.3 | 7B | 32K | Balanced | 8GB RAM |
| **Mistral AI** | Mistral 8x7B MoE | 47B (active 13B) | 32K | High IQ | 16GB+ RAM |
| **Mistral AI** | Mixtral 8x22B | 141B (active 39B) | 64K | Expert-level | 48GB+ RAM |
| **Google** | Gemma 2 | 2B, 9B, 27B | 8K | Research/Commercial | 4GB+ RAM |
| **Google** | Gemma | 2B, 7B | 8K | Lightweight | 4GB+ RAM |
| **Other Open** | Qwen 2.5 | 0.5B to 72B | 32K-128K | Multilingual | *Varies by size* |
| **Other Open** | Falcon | 7B, 40B | 2K | Commercial use | *Varies by size* |
| **Other Open** | Mosaic MPT-7B | 7B, 30B | 65K-84K | Long context | *Varies by size* |
| **Other Open** | StableLM 2 | 1.6B, 12B | 4K | Stability AI | *Varies by size* |
| **Other Open** | OLMo 7B | 7B | 2K | Pure open (data+weights) | *Varies by size* |
| **Other Open** | BLOOM | 1B to 176B | Various | Multilingual (56 langs) | *Varies by size* |
| --- | --- | --- | --- | --- | --- |
| **Quantized** | Llama 3.1 8B Q4_K_M | ~8B (Quantized) | N/A | Excellent quality, GGUF Q4 | ~4.5GB (requires 6GB total) |
| **Quantized** | Mistral 7B Q5_K_M | ~7B (Quantized) | N/A | Very Good quality, GGUF Q5 | ~4.8GB (requires 7GB total) |
| **Quantized** | Phi-3.5 Mini Q4 | N/A (Quantized) | N/A | Good quality, GGUF Q4 | ~2.2GB (requires 4GB total) |
| **Quantized** | Gemma 2 9B Q4 | ~9B (Quantized) | N/A | Very Good quality, GGUF Q4 | ~5.2GB (requires 8GB total) |

# 1. 🔍 Retrieval and Formatting (The Input Stage)
This first step takes your original question and uses it to look up relevant evidence.

{"context": retriever | format_docs, "question": RunnablePassthrough()}
{"context": ..., "question": ...}: This creates a dictionary that acts as the required input format for the prompt template.

# "question": RunnablePassthrough():

Function: Takes the user's initial query and passes it through directly, unchanged.

Simple Word: This is just the original question.

# "context": retriever | format_docs: This is the core Retrieval-Augmentation step.

# retriever
: Your configured search tool (FAISS index + Sentence-Transformer embeddings). It takes the question, finds the top relevant text chunks from your medical data, and passes them along.

# format_docs:
This is a simple function that takes the individual text chunks returned by the retriever and combines them into one single, long string of text. This becomes the evidence (the "context") for the LLM.
# 2. 📝 Augmentation (The Prompt Stage)
The output from the first stage (the dictionary containing the context and question) is fed into the prompt template.

# |  RAG_PROMPT
RAG_PROMPT: This is your carefully designed Prompt Template. It performs the crucial Augmentation step by taking the retrieved evidence and embedding it directly into the instructions for the LLM.

Simple Word: This creates the final, complete instruction: Answer the following Question based ONLY on this Context.
#  3. 🧠 Generation and Cleanup (The Output Stage)
The final, augmented prompt is passed to the language model to generate the answer, which is then cleaned up for the user.

| llm
| StrOutputParser()
| llm:

Function: This is your loaded language model (Flan-T5 Small). It reads the entire augmented prompt and generates the medical answer.

Simple Word: This is the answer generator.

# | StrOutputParser():

Function: Converts the LLM's raw output object (which often contains extra metadata) into a clean, simple Python string that can be easily displayed.

Simple Word: The cleaner that provides the final, readable text.
# 🎯 Conclusion: Transition to High-Performance LLMs
The initial performance assessment, utilizing the resource-efficient google/flan-t5-small model, successfully validated the Retrieval-Augmented Generation (RAG) pipeline's architecture, demonstrating its ability to accurately retrieve relevant medical context from the indexed knowledge base. However, the small model exhibited critical limitations in generation, resulting in verbose, repeated, and noisy outputs that failed to meet the required quality standards for a reliable medical question-answering system.
# 📈 Impact of LLM Upgrade
The transition to a state-of-the-art, high-level LLM (such as GPT-4 or Gemini) addresses all previous shortcomings and is justified by the following observed performance gains:

# 1. Precision and Accuracy:
  The advanced LLM demonstrates a near-perfect ability to adhere to the prompt constraints, specifically the crucial instruction to "answer ONLY based on the context." This eliminates previous issues like hallucination (e.g., mentioning "retinal detachment" when not in the source).

# 2 Summarization and Conciseness: 
The new model handles the complexity of the retrieved context seamlessly. It successfully summarizes long chunks into crisp, direct answers, eliminating the redundant, repeated text and irrelevant metadata (like video instructions) observed with Flan-T5 Small.

# 3. Robust Extraction:
Complex, multi-part queries (e.g., "List the specific treatments...") are now answered with clean, structured lists, demonstrating superior information extraction capabilities from dense source text.

# 4. Improved User Experience:
The final output is no longer a debugging challenge but a reliable, professional, and trustworthy medical response, crucial for a production-ready application dealing with sensitive health information.
