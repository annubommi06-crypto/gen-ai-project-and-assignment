# Modern Vector Search & Information Retrieval Pipelines

Modern information retrieval has evolved from lexical keyword matching (TF-IDF, BM25) to dense semantic representations powered by neural transformers.

## Key Architectural Principles

| Metric | Lexical Search (BM25) | Dense Vector Search (HNSW) | Hybrid Search |
|---|---|---|---|
| Representation | Sparse word frequency vectors | Dense continuous embeddings | Weighted reciprocal rank fusion |
| Synonym Handling | Requires manual thesaurus | Native semantic understanding | Robust across both vocabulary gaps & exact IDs |
| Query Latency | Sub-millisecond | 2 - 10 milliseconds | 5 - 15 milliseconds |
| Storage Footprint | Low (Inverted Index) | Moderate to High (Vector Index) | Combined Index |

### Implementation Considerations

When building vector-based search:
1. Always normalize your embeddings so that dot product matches cosine similarity.
2. Store rich metadata alongside your vectors to support structured pre-filtering.
3. Test multiple chunk sizes (e.g., 200, 500, 1000 characters) to identify the optimal context window for your embedding model.

