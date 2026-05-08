"""
50 QA pairs with ground-truth answers for RAGAS evaluation.
Each entry has "question" and "reference" fields.
"""

QA_PAIRS = [
    {
        "question": "What are the three main types of machine learning?",
        "reference": "The three main types of machine learning are supervised learning, unsupervised learning, and reinforcement learning. Supervised learning uses labeled datasets to learn input-output mappings. Unsupervised learning discovers hidden patterns in unlabeled data. Reinforcement learning trains an agent through rewards and penalties based on actions taken in an environment.",
    },
    {
        "question": "What is overfitting in machine learning?",
        "reference": "Overfitting occurs when a machine learning model learns the training data too well, including noise and random fluctuations, at the expense of generalization to unseen data. An overfitted model captures idiosyncratic patterns that exist only in the training set but do not reflect the true underlying relationship. Symptoms include high training accuracy but low validation/test accuracy.",
    },
    {
        "question": "Explain the bias-variance tradeoff.",
        "reference": "High bias means underfitting; high variance means overfitting. The optimal model balances both to minimize total error. Total error = Bias² + Variance + Irreducible Noise. Simple models (linear regression, naive Bayes) tend to have high bias but low variance. Complex models (deep neural networks) tend to have low bias but high variance.",
    },
    {
        "question": "How does regularization prevent overfitting?",
        "reference": "Regularization prevents overfitting by adding a penalty term to the model's loss function that discourages overly complex or large parameter values. L1 (Lasso) adds the sum of absolute parameter values, encouraging sparsity. L2 (Ridge) adds the sum of squared parameter values, shrinking all parameters toward zero. Dropout randomly sets neuron activations to zero during training, preventing co-adaptation.",
    },
    {
        "question": "What is cross-validation?",
        "reference": "Cross-validation splits data into multiple folds to estimate model performance more reliably. K-fold cross-validation splits data into k equal folds, trains k times (k-1 folds for training, 1 for validation), and averages performance. Stratified K-Fold preserves the class distribution in each fold. Nested cross-validation is used for hyperparameter tuning and model selection to prevent data leakage.",
    },
    {
        "question": "What is backpropagation?",
        "reference": "Backpropagation (backward propagation of errors) is the algorithm used to train neural networks by computing gradients of the loss function with respect to each weight. It works in two passes: a forward pass computes predictions layer by layer, then the backward pass propagates gradients from the output layer back through the network using the chain rule of calculus to compute weight gradients, which are used to update weights via gradient descent.",
    },
    {
        "question": "What are Convolutional Neural Networks primarily used for?",
        "reference": "Convolutional Neural Networks (CNNs) are primarily used for image classification, object detection, image segmentation, face recognition, scene understanding, video analysis, and medical imaging. CNNs automatically learn spatial hierarchies of features from grid-like data (primarily images) through convolutional layers, activation functions, and pooling layers, followed by fully connected layers.",
    },
    {
        "question": "How do LSTM networks address the vanishing gradient problem?",
        "reference": "LSTMs address the vanishing gradient problem by introducing a memory cell (cell state) that acts as a conveyor belt through time, with information added or removed via carefully controlled gates (forget gate, input gate, output gate). The additive gradient flow through the cell state enables gradients to flow more easily across long sequences, preventing them from exponentially shrinking as they propagate backward through time.",
    },
    {
        "question": "What activation functions are commonly used in neural networks?",
        "reference": "Commonly used activation functions include: Sigmoid (output range 0-1, used in output layers for binary classification), Tanh (output range -1 to 1, zero-centered but suffers vanishing gradients), ReLU (max(0, x), most widely used, computationally efficient and reduces vanishing gradients), Leaky ReLU (max(0.01x, x), solves dying ReLU problem), Softmax (used for multi-class classification output), and GELU (used in Transformers, better than ReLU for attention-based models).",
    },
    {
        "question": "What is the role of pooling layers in CNNs?",
        "reference": "Pooling layers reduce spatial dimensions (width and height) of feature maps while retaining important information. Max pooling takes the maximum value in each window, providing translation invariance and reducing sensitivity to exact pixel positions. Average pooling computes the average value in each window. Benefits include dimensionality reduction, translation invariance, control of overfitting, and hierarchical feature combination.",
    },
    {
        "question": "What is the transformer architecture?",
        "reference": "The Transformer architecture is a neural network based entirely on self-attention mechanisms, eliminating recurrence and convolutions. It consists of an encoder with N identical layers (multi-head self-attention + feed-forward network) and a decoder with N identical layers (masked multi-head self-attention + encoder-decoder attention + feed-forward network). Key advantages include full parallelization, O(1) path length for long-range dependencies, and interpretability via attention weights.",
    },
    {
        "question": "What are word embeddings?",
        "reference": "Word embeddings are dense vector representations of words in a continuous vector space where semantically similar words are mapped to nearby points. They capture semantic meaning, syntactic relationships, and contextual nuances. Examples include Word2Vec (skip-gram and CBOW architectures, static embeddings), GloVe (global co-occurrence statistics), FastText (subword information), ELMo (contextual embeddings), and BERT (bidirectional contextual embeddings using Transformer encoder).",
    },
    {
        "question": "What is transfer learning in NLP?",
        "reference": "Transfer learning in NLP is the practice of pre-training a large language model on a vast corpus of text data using self-supervised objectives, then fine-tuning it on a specific downstream task with a smaller labeled dataset. The two-stage paradigm consists of pre-training (using masked language modeling or next token prediction on large unlabeled data) followed by fine-tuning (training on task-specific labeled data with minimal task-specific parameters).",
    },
    {
        "question": "How does BERT handle language understanding?",
        "reference": "BERT handles language understanding through bidirectional context and masked language modeling. Unlike causal language models that only look left, BERT reads the entire sequence simultaneously and uses a Masked Language Model (MLM) objective that randomly masks tokens and trains the model to predict them using context from both left and right. BERT uses only the Transformer encoder and is pre-trained on BookCorpus and Wikipedia, then fine-tuned on downstream tasks with task-specific heads.",
    },
    {
        "question": "What is self-attention in transformers?",
        "reference": "Self-attention (scaled dot-product attention) allows each token in a sequence to attend to all other tokens. It works by projecting each input token into Query, Key, and Value vectors, computing attention scores as Q·K^T / sqrt(dk), applying softmax to get attention weights, and computing the weighted sum of values. Multi-head attention runs multiple attention mechanisms in parallel, allowing the model to jointly attend to information from different representation subspaces.",
    },
    {
        "question": "What is GPT and how is it trained?",
        "reference": "GPT (Generative Pre-trained Transformer) is a decoder-only Transformer language model pre-trained on massive text data using next-token prediction (causal language modeling), where the model learns to predict the next token given left context only. GPT-3.5+ and GPT-4 use RLHF (Reinforcement Learning from Human Feedback): supervised fine-tuning on demonstration data, reward model training on human comparisons, and PPO fine-tuning to maximize predicted rewards.",
    },
    {
        "question": "What is instruction tuning?",
        "reference": "Instruction tuning is a fine-tuning technique that improves a language model's ability to follow natural language instructions. It involves collecting instruction-response pairs (datasets like FLAN, Alpaca, Dolly), fine-tuning the pre-trained model using standard supervised fine-tuning, and optionally applying RLHF. Instruction tuning bridges the gap between next-token prediction and helpful instruction-following, enabling models like InstructGPT and Alpaca.",
    },
    {
        "question": "What is RLHF?",
        "reference": "RLHF (Reinforcement Learning from Human Feedback) is a three-stage pipeline for aligning language models with human preferences. Stage 1: Supervised Fine-Tuning (SFT) on curated demonstration data. Stage 2: Reward Model training on human comparisons to predict which response humans prefer. Stage 3: RL fine-tuning using PPO (Proximal Policy Optimization) to maximize rewards predicted by the reward model, with a KL divergence penalty to prevent drifting too far from the SFT baseline.",
    },
    {
        "question": "What is chain-of-thought prompting?",
        "reference": "Chain-of-thought (CoT) prompting is a prompting technique that elicits explicit intermediate reasoning steps from a language model before producing the final answer. By including 'Let's think step by step' or providing reasoning examples, the model generates intermediate steps, dramatically improving performance on multi-step arithmetic, symbolic manipulation, and commonsense reasoning tasks. Zero-shot CoT uses the phrase without examples; few-shot CoT includes examples.",
    },
    {
        "question": "What is the context length of GPT-4?",
        "reference": "GPT-4 Turbo supports a context window of 128,000 tokens (approximately 100,000 words or about 300 pages of text). Earlier GPT models had smaller context windows: GPT-3 had 4,096 tokens, GPT-3.5 had 4,096 to 16,384 tokens, and the original GPT-4 had 8,192 tokens. Longer context enables in-context learning with many examples, processing entire documents, and reduces the need for retrieval-augmented generation in some cases.",
    },
    {
        "question": "What is Retrieval-Augmented Generation?",
        "reference": "Retrieval-Augmented Generation (RAG) is a hybrid architecture that combines a retrieval system with a generative language model. It allows models to retrieve relevant documents from an external knowledge base (non-parametric memory) at inference time. RAG solves knowledge cut-off issues, reduces hallucination by grounding responses in retrieved evidence, provides transparency through source attribution, and enables domain adaptation with proprietary data without retraining.",
    },
    {
        "question": "What are the main components of a RAG pipeline?",
        "reference": "The main components of a RAG pipeline are: 1) Document Loader (reads from PDFs, HTML, databases), 2) Text Splitter/Chunker (splits documents into chunks for embedding), 3) Embedding Model (converts text to dense vectors), 4) Vector Database (FAISS, Pinecone, etc. for similarity search), 5) Retriever (finds relevant chunks given a query), 6) Prompt Template (injects context into LLM prompt), 7) Language Model (generates the final response), and 8) Output Parser (post-processes structured output).",
    },
    {
        "question": "What is dense retrieval?",
        "reference": "Dense retrieval uses dense vector embeddings to represent both queries and documents, enabling semantic similarity search. Both the query and documents are encoded into dense vectors using a neural embedding model (encoder). During retrieval, the query vector is compared against document vectors using cosine similarity or dot product, and top-k documents are returned. Dense retrieval captures semantic relationships (synonyms) but struggles with exact lexical matching.",
    },
    {
        "question": "Why is chunking strategy important in RAG?",
        "reference": "Chunking strategy determines what information is retrieved and presented to the LLM. Key dimensions: chunk size (too small misses context, too large dilutes signal; optimal is 500-1000 tokens), chunk overlap (ensures boundary information is not lost), semantic vs fixed-size chunking, document structure awareness (respect headers, tables), and parent document chunking (small chunks for retrieval, parent chunks for generation).",
    },
    {
        "question": "What advanced RAG techniques exist beyond basic retrieval?",
        "reference": "Advanced RAG techniques include: Hybrid Search (combining dense and sparse BM25 retrieval), Query Expansion/Decomposition (rephrasing or decomposing complex questions), Reranking (using cross-encoder to refine initial results), Contextual Chunk Enrichment (LLM-generated summaries for chunks), Self-Query/Adaptive Retrieval (LLM decides when to retrieve), Iterative Retrieval (multi-round retrieval refining context), Corrective RAG (assess and retry retrieval on low quality), and Graph RAG (knowledge graph for entity organization).",
    },
    {
        "question": "What are vector databases used for?",
        "reference": "Vector databases store, index, and search high-dimensional vector embeddings efficiently. They are the backbone of AI retrieval systems, enabling sub-linear search complexity (O(log n)) compared to naive O(n) exhaustive search. They store embeddings with optional metadata (text content, document ID, source URL). Key indexing algorithms include HNSW (Hierarchical Navigable Small World), IVF (Inverted File Index), and PQ (Product Quantization). Examples: FAISS, Pinecone, Weaviate, Milvus, Qdrant, ChromaDB.",
    },
    {
        "question": "What is FAISS?",
        "reference": "FAISS (Facebook AI Similarity Search) is an open-source library for efficient dense vector similarity search and clustering. It handles billion-scale vector databases with sub-linear search complexity. Index types include Flat (brute-force, exact), IVF (Inverted File Index, clusters vectors for faster search), HNSW (Hierarchical Navigable Small World, graph-based, excellent quality-speed tradeoff), and PQ (Product Quantization, compression for large-scale). LangChain provides the FAISS class in langchain_community.vectorstores.",
    },
    {
        "question": "How do text embeddings capture semantic meaning?",
        "reference": "Text embeddings capture semantic meaning by representing text as dense vectors in a continuous space where similar meanings are encoded as nearby vectors. They are trained on large text corpora using self-supervised objectives (CBOW, Skip-gram, Masked Language Modeling, Contrastive Learning). The vector space encodes semantic similarity (cosine similarity between vectors), analogy relationships (king - man + woman ≈ queen), contextual meaning (BERT-style), and topic clustering.",
    },
    {
        "question": "What is HNSW?",
        "reference": "HNSW (Hierarchical Navigable Small World) is a graph-based ANN indexing algorithm enabling fast, high-quality similarity search. It builds a multi-layer graph where vectors are assigned to layers based on exponential probability distribution. Layer 0 contains all vectors with dense connections; upper layers have sparse long-range connections acting as highways. Search is O(log n) per query with high accuracy (>95% recall). No training required. Key parameters: M (connections per node), efConstruction (build quality), efSearch (query quality).",
    },
    {
        "question": "What is hybrid search in vector databases?",
        "reference": "Hybrid search combines dense retrieval (semantic similarity via embeddings) with sparse retrieval (BM25 keyword matching) to leverage their complementary strengths. Dense retrieval handles semantic relationships; sparse retrieval handles exact keyword matches. Combined scores outperform either method alone. Common fusion methods include Reciprocal Rank Fusion (RRF), weighted sum, and ColBERT-style late interaction. Implemented in dedicated databases (Weaviate, Qdrant) or by combining FAISS with a separate BM25 implementation.",
    },
    {
        "question": "What is LangChain?",
        "reference": "LangChain is an open-source framework for building LLM-powered applications. It provides modular components for prompts, models, tools, memory, and retrieval, with flexibility to swap providers without rewriting code. Core modules include Models (LLMs, chat models, embeddings), Prompts (templates, few-shot selectors), Chains (composable sequences), Indexes (loaders, splitters, vector stores), Agents (LLM-driven tool use), Memory (conversation history), and Callbacks (observability). LangChain Expression Language (LCEL) enables declarative chain composition.",
    },
    {
        "question": "What is LangChain Expression Language (LCEL)?",
        "reference": "LCEL (LangChain Expression Language) is a declarative pipe-based syntax for composing LangChain chains using the pipe operator (|). Key features include streaming (token-by-token output), async support (native async/await), parallel execution (RunnableParallel), batch processing (.batch()), and Runnable interface (every component implements Runnable with .invoke(), .stream(), .batch(), .ainvoke() methods). Dict inputs are automatically routed to matching keys in the next component.",
    },
    {
        "question": "What is LangGraph?",
        "reference": "LangGraph is LangChain's extension for building stateful, cyclic workflows for agentic systems where the model loops back on itself. Unlike standard LangChain chains (acyclic DAGs), LangGraph supports cycles for iterative workflows. Core concepts: State (shared dict persisting across steps), Nodes (functions performing a step), Edges (control flow with conditional routing), and Cycles (edges pointing backward for loops). Use cases include ReAct agents, multi-agent systems, self-refinement, and conversational RAG.",
    },
    {
        "question": "What memory types does LangChain support?",
        "reference": "LangChain supports several memory types: ConversationBufferMemory (stores all messages, problematic for long conversations), ConversationBufferWindowMemory (keeps only last k messages), ConversationTokenBufferMemory (counts tokens instead of messages), ConversationSummaryMemory (summarizes history into compact form), ConversationSummaryBufferMemory (keeps recent messages verbatim, summarizes older), VectorStore-Backed Memory (stores history in vector DB for semantic search), Entity Memory (extracts and stores entities), and Hybrid Memory (combines multiple types).",
    },
    {
        "question": "What are LangChain retrievers?",
        "reference": "LangChain retrievers take a query and return relevant documents from a data source, implementing the BaseRetriever protocol with .get_relevant_documents(query) method. Built-in retrievers include VectorStore Retriever (wraps vector store), Ensemble Retriever (fuses multiple retrievers via RRF), Parent Document Retriever (two-stage: child chunks for retrieval, parent chunks for generation), MultiQuery Retriever (LLM generates query variations), Contextual Compression Retriever (extracts relevant portions), and Self-Query Retriever (LLM parses structured filters).",
    },
    {
        "question": "What is LangSmith?",
        "reference": "LangSmith is a comprehensive platform for debugging, testing, evaluating, and monitoring LLM applications. Core capabilities include: Tracing (records every step with inputs, outputs, metadata, latency, token counts), Datasets (curated input-output pairs for evaluation), Evaluation (LLM-as-judge metrics), Prompt Hub (versioned, deployed prompts), Monitoring (latency, cost, error rates dashboard), and Feedback (user ratings and annotations). It is part of the LangChain ecosystem providing enterprise-grade observability.",
    },
    {
        "question": "What information do LangSmith traces capture?",
        "reference": "LangSmith traces capture: Run metadata (unique ID, timestamps, duration, parent-child hierarchy, tags), Input (exact input to each step), Output (exact output from each step), Token usage (input/output/total tokens per LLM call), Latency (time per component, identifies bottlenecks), Model information (model name, temperature), Errors (exceptions and error messages), and Feedback (user ratings and annotations). Traces form a tree hierarchy enabling drill-down from chain-level to component-level.",
    },
    {
        "question": "What is the LangSmith Prompt Hub?",
        "reference": "LangSmith Prompt Hub is a managed prompt registry for versioned, shared, and deployed prompts. Key features: Push prompts (save from code with name, version, description, tags), Pull prompts (retrieve latest or specific version), Versioning (each push creates a new version with history), Collaboration (team sharing), Deployment (centralized prompt management), and Prompt Playground (interactive testing UI). API: client.push_prompt(name, object, description) and client.pull_prompt(name).",
    },
    {
        "question": "How does LangSmith help monitor production LLM applications?",
        "reference": "LangSmith monitoring includes: Latency monitoring (response time per chain and component), Cost tracking (token usage and cost by model/time period), Error rate monitoring (tracks exceptions with anomaly alerts), Model comparison (metrics across different models), Feedback loop (user thumbs up/down and ratings), Dataset evaluation (run evaluation on traffic or curated datasets), Custom dashboards (combine metrics), and Alerting (configure threshold alerts). Enables data-driven decisions on model selection and optimization.",
    },
    {
        "question": "What are LangSmith datasets used for?",
        "reference": "LangSmith datasets are curated collections of inputs (and optionally expected outputs) for systematic evaluation. Each entry is a RunExample with input fields and optional reference fields. Use cases: Benchmarking (establish baseline performance), Regression testing (detect performance changes after code updates), A/B evaluation (compare prompt or model versions), Fine-tuning data curation (identify low-quality examples from production traces), and Continuous evaluation (periodic nightly/weekly runs to track quality over time).",
    },
    {
        "question": "What is RAGAS?",
        "reference": "RAGAS (RAG Assessment) is an open-source framework for evaluating RAG pipelines using automated, reference-free metrics. It uses LLM-as-judge to assess retrieval quality and generation quality. Key metrics: Faithfulness (factual accuracy of generated answer wrt retrieved context), Answer Relevancy (how directly the answer addresses the question), Context Recall (retrieved context covers needed information, requires reference), and Context Precision (retrieved context contains relevant information).",
    },
    {
        "question": "How does RAGAS compute faithfulness?",
        "reference": "RAGAS computes faithfulness in three steps: 1) Generate statements (LLM parses the generated answer into individual factual claims), 2) Verify each statement (LLM judges whether each claim is supported by the retrieved context), 3) Compute score (faithfulness = number of supported statements / total statements, score between 0 and 1). A score of 1.0 means all claims are supported by context; 0.0 means none are supported. Target for lab: ≥ 0.8.",
    },
    {
        "question": "What is answer relevancy in RAGAS?",
        "reference": "Answer Relevancy in RAGAS measures how directly and comprehensively the generated answer addresses the user's question. It works in two steps: 1) Generate questions (LLM generates potential questions the answer would appropriately answer), 2) Compute similarity (embed the generated questions and compute cosine similarity to the original question, average similarity is the score). High relevancy means the answer is on-topic, focused, and contains directly related information without unnecessary tangents.",
    },
    {
        "question": "What is context recall in RAGAS?",
        "reference": "Context Recall in RAGAS measures whether the retrieved context contains all the information needed to answer the question. It requires a ground truth reference answer. Steps: 1) Extract statements from ground truth, 2) For each claim, LLM judges whether it can be attributed to the retrieved context (Attributed or Not Attributed), 3) Compute score (context recall = attributed claims / total claims in ground truth). Low context recall indicates retrieval gaps, regardless of LLM quality.",
    },
    {
        "question": "What inputs does RAGAS evaluation require?",
        "reference": "RAGAS evaluation requires SingleTurnSample objects containing: user_input (the question/query), response (generated answer from the RAG pipeline), retrieved_contexts (list of text chunks retrieved), and reference (ground truth reference answer, required for context_recall but optional for other metrics). These are wrapped in EvaluationDataset for evaluation. The evaluate() function takes dataset and metrics, plus llm and embeddings for LLM-as-judge scoring.",
    },
    {
        "question": "What is Guardrails AI?",
        "reference": "Guardrails AI is an open-source framework for adding validation, correction, and safety checks to LLM outputs. It provides a declarative rule-based system for ensuring outputs meet quality and safety standards. The main entry point is the Guard class, which defines a validation pipeline. Validators are Python classes decorated with @register_validator, implementing validate(value, metadata) to return PassResult or FailResult. on_fail actions include EXCEPT, REJECT, FIX, WARN, NOOP.",
    },
    {
        "question": "What is PII and why is it important to detect in LLM responses?",
        "reference": "PII (Personally Identifiable Information) is any data that can identify, contact, or locate a specific individual. Types include email addresses, phone numbers, Social Security Numbers, credit card numbers, physical addresses, names, dates of birth, and IP addresses. PII detection is important for: privacy regulations (GDPR, CCPA compliance), data minimization (preventing LLM memorization/repetition of PII), user safety (protecting individuals from information exposure), and enterprise trust (guarantees that LLM applications won't leak sensitive data).",
    },
    {
        "question": "What does structured output validation ensure?",
        "reference": "Structured output validation ensures LLM responses conform to a specific format, schema, or constraints. It ensures downstream parsing reliability (e.g., JSON output parsing), output consistency across responses, and error handling with auto-repair. Common issues include markdown code fences (```json blocks), single quotes instead of double quotes (JSON requires double quotes), trailing commas, Python dict literals, missing quotes on keys, and extra explanatory text. Auto-repair tries stripping fences, fixing quotes, removing trailing commas, and retrying JSON parse.",
    },
    {
        "question": "What is Constitutional AI?",
        "reference": "Constitutional AI (CAI) is a research direction from Anthropic that trains AI systems to be helpful, harmless, and honest using guiding principles (a constitution) rather than relying solely on human feedback. The process: Critique (model identifies ways response might violate principles), Revision (model revises to address issues), and RLAIF (use model judgments as reward signal instead of human preferences). Benefits: scalable (reduces human labels needed), transparent (principles are explicit), adaptable (new principles can be added), and self-consistent.",
    },
    {
        "question": "What are common AI safety concerns with LLMs?",
        "reference": "Common AI safety concerns include: Hallucination (plausible but factually incorrect information, dangerous in high-stakes domains), Misinformation (convincing false narratives or fake news), Toxicity (offensive, hateful, violent content), Privacy violations (exposing personal information from training data), Jailbreaking (bypassing safety measures through creative prompts), Prompt injection (malicious instructions in user inputs), Sycophancy (agreeing with user beliefs even when wrong), Bias and fairness (amplifying societal biases), Overreliance (users trusting AI without verification), and Model robustness (sensitivity to small input changes).",
    },
]
