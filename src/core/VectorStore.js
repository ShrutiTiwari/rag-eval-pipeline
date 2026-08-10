/**
 * VectorStore - Handles document embeddings and similarity search.
 * The embedding model is supplied via an EmbeddingProvider; swap it with
 * a one-line config change (EMBEDDING_PROVIDER=openai) rather than touching this file.
 */
const { createEmbeddingProvider } = require('../providers/EmbeddingProvider');

class VectorStore {
  /**
   * @param {import('../providers/EmbeddingProvider').EmbeddingProvider} [embeddingProvider]
   *   Optional provider instance. Defaults to createEmbeddingProvider().
   */
  constructor(embeddingProvider = null) {
    this.embeddings = [];
    this.embeddingProvider = embeddingProvider || createEmbeddingProvider();
  }

  /**
   * Generate embeddings for document chunks
   * @param {Array<Object>} chunks - Array of document chunks
   * @param {Object} metadata - Document metadata
   * @returns {Promise<Array<Object>>} Array of embeddings with metadata
   */
  async generateEmbeddings(chunks, metadata = {}) {
    if (!chunks || !Array.isArray(chunks) || chunks.length === 0) {
      throw new Error('Chunks must be a non-empty array');
    }

    console.log(`Generating embeddings for ${chunks.length} chunks...`);

    const texts = chunks.map(chunk => chunk.content);
    const vectors = await this.embeddingProvider.embed(texts);

    const embeddings = chunks.map((chunk, i) => ({
      id: `${metadata.filename || 'doc'}_chunk_${chunk.index || i}`,
      content: chunk.content,
      embedding: vectors[i],
      metadata: {
        ...metadata,
        chunkIndex: chunk.index || i,
        startChar: chunk.startChar,
        endChar: chunk.endChar,
        length: chunk.length || chunk.content.length,
        section: chunk.section || null,
        strategy: chunk.strategy || null
      },
      createdAt: new Date().toISOString()
    }));

    console.log(`Generated ${embeddings.length} embeddings successfully`);
    return embeddings;
  }

  /**
   * Add embeddings to the vector store
   * @param {Array<Object>} embeddings - Array of embeddings
   */
  addEmbeddings(embeddings) {
    if (!embeddings || !Array.isArray(embeddings)) {
      throw new Error('Embeddings must be an array');
    }

    this.embeddings.push(...embeddings);
    console.log(`Added ${embeddings.length} embeddings to vector store (Total: ${this.embeddings.length})`);
  }

  /**
   * Generate embedding for a query string
   * @param {string} query - Search query
   * @returns {Promise<Array<number>>} Query embedding vector
   */
  async generateQueryEmbedding(query) {
    if (!query || typeof query !== 'string') {
      throw new Error('Query must be a non-empty string');
    }

    return this.embeddingProvider.embedOne(query.trim());
  }

  /**
   * Perform similarity search using cosine similarity
   * @param {string} query - Search query
   * @param {number} topK - Number of top results to return
   * @param {number} threshold - Minimum similarity threshold (0-1)
   * @returns {Promise<Array<Object>>} Top similar chunks with scores
   */
  async similaritySearch(query, topK = 5, threshold = 0.1) {
    if (this.embeddings.length === 0) {
      throw new Error('No embeddings found in vector store');
    }

    console.log(`Searching for: "${query}" (topK=${topK}, threshold=${threshold})`);

    // Generate query embedding
    const queryEmbedding = await this.generateQueryEmbedding(query);

    // Calculate similarities
    const similarities = this.embeddings.map(item => ({
      ...item,
      similarity: this.cosineSimilarity(queryEmbedding, item.embedding)
    }));

    // Filter by threshold and sort by similarity
    const filteredResults = similarities
      .filter(item => item.similarity >= threshold)
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, topK);

    console.log(`Found ${filteredResults.length} results above threshold ${threshold}`);

    return filteredResults.map(result => ({
      id: result.id,
      content: result.content,
      similarity: result.similarity,
      metadata: result.metadata
    }));
  }

  /**
   * Hybrid search: weighted combination of cosine similarity and BM25 keyword score.
   * Solves the problem where structured content (piece titles, grade numbers, scale names)
   * scores low on cosine similarity because the vocabulary doesn't match natural language
   * queries — BM25 catches exact keyword matches that semantic search misses.
   *
   * Final score = alpha * cosineScore + (1 - alpha) * bm25Score  (both normalised to 0-1)
   *
   * @param {string} query     - Search query
   * @param {number} topK      - Number of results to return
   * @param {number} threshold - Minimum hybrid score threshold (0-1)
   * @param {number} alpha     - Weight for cosine (0=pure BM25, 1=pure cosine, default 0.5)
   * @returns {Promise<Array<Object>>} Top results with hybrid score and component scores
   */
  async hybridSearch(query, topK = 5, threshold = 0.1, alpha = 0.5) {
    if (this.embeddings.length === 0) {
      throw new Error('No embeddings found in vector store');
    }

    console.log(`Hybrid search for: "${query}" (topK=${topK}, alpha=${alpha})`);

    // ── Step 1: cosine similarity scores ──────────────────────────────────
    const queryEmbedding = await this.generateQueryEmbedding(query);
    const cosineScores = this.embeddings.map(item =>
      this.cosineSimilarity(queryEmbedding, item.embedding)
    );

    // Normalise cosine scores to 0-1 (they already are, but clamp for safety)
    const maxCosine = Math.max(...cosineScores, 1e-9);
    const normCosine = cosineScores.map(s => s / maxCosine);

    // ── Step 2: BM25 scores ────────────────────────────────────────────────
    const bm25Scores = this.bm25Score(query);

    // Normalise BM25 to 0-1
    const maxBm25 = Math.max(...bm25Scores, 1e-9);
    const normBm25 = bm25Scores.map(s => s / maxBm25);

    // ── Step 3: weighted combination ──────────────────────────────────────
    const results = this.embeddings.map((item, i) => ({
      id: item.id,
      content: item.content,
      metadata: item.metadata,
      similarity: alpha * normCosine[i] + (1 - alpha) * normBm25[i], // hybrid score
      cosineScore: cosineScores[i],   // raw scores kept for scorecard display
      bm25Score: bm25Scores[i]
    }));

    const filtered = results
      .filter(r => r.similarity >= threshold)
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, topK);

    console.log(`Hybrid search found ${filtered.length} results (alpha=${alpha})`);
    return filtered;
  }

  /**
   * BM25 scoring for all embeddings against a query.
   * BM25 is a probabilistic keyword ranking function — it rewards chunks that
   * contain query terms frequently (TF) relative to their length, discounted by
   * how common the term is across all chunks (IDF).
   *
   * Parameters k1=1.5, b=0.75 are the standard BM25 defaults.
   *
   * @param {string} query
   * @returns {Array<number>} BM25 score per embedding (same order as this.embeddings)
   */
  bm25Score(query) {
    const k1 = 1.5;
    const b  = 0.75;

    // Common English stopwords — high-frequency words that carry no retrieval signal.
    // Removed from both query and document tokens so they don't inflate BM25 scores
    // for irrelevant prose chunks (e.g. "practice" matching "Purchasing these books is
    // not a requirement" because both contain "for", "the", "a").
    const STOPWORDS = new Set([
      'a','an','the','and','or','but','in','on','at','to','for','of','with',
      'by','from','is','are','was','were','be','been','being','have','has',
      'had','do','does','did','will','would','could','should','may','might',
      'i','you','he','she','it','we','they','what','which','who','this','that',
      'these','those','not','no','so','if','as','up','out','about','into','than',
      'then','there','can','just','more','also','any','all','its','their','my',
    ]);

    // Tokenise: lowercase, split on non-word chars, drop stopwords.
    // Keep numeric tokens like "7" in "Grade 7" — critical for grade-specific queries.
    const tokenise = text => text.toLowerCase().split(/\W+/)
      .filter(t => t.length > 0 && !STOPWORDS.has(t));

    const queryTerms = tokenise(query);
    if (queryTerms.length === 0) return this.embeddings.map(() => 0);

    const docs = this.embeddings.map(e => tokenise(e.content));
    const N = docs.length;
    const avgDl = docs.reduce((s, d) => s + d.length, 0) / N;

    // IDF per query term: log((N - df + 0.5) / (df + 0.5) + 1)
    const idf = {};
    for (const term of queryTerms) {
      if (idf[term] !== undefined) continue;
      const df = docs.filter(d => d.includes(term)).length;
      idf[term] = Math.log((N - df + 0.5) / (df + 0.5) + 1);
    }

    // Build bigrams from query for phrase matching (e.g. "grade 7" as a unit)
    const queryBigrams = [];
    for (let i = 0; i < queryTerms.length - 1; i++) {
      queryBigrams.push(queryTerms[i] + '_' + queryTerms[i + 1]);
    }

    // Score each document
    return docs.map((tokens, docIdx) => {
      const dl = tokens.length;
      const tf = {};
      for (const t of tokens) tf[t] = (tf[t] || 0) + 1;

      let score = 0;
      for (const term of queryTerms) {
        const termFreq = tf[term] || 0;
        score += idf[term] * (termFreq * (k1 + 1)) / (termFreq + k1 * (1 - b + b * dl / avgDl));
      }

      // Phrase bonus: if the document contains adjacent query bigrams (e.g. "grade 7"),
      // add a fixed bonus per matching bigram. This lifts chunks that contain the exact
      // phrase above chunks that only match individual tokens.
      if (queryBigrams.length > 0) {
        const docBigrams = new Set();
        for (let i = 0; i < tokens.length - 1; i++) {
          docBigrams.add(tokens[i] + '_' + tokens[i + 1]);
        }
        for (const bigram of queryBigrams) {
          if (docBigrams.has(bigram)) score += 2.0;
        }
      }

      return score;
    });
  }

  /**
   * Calculate cosine similarity between two vectors
   * @param {Array<number>} vecA - First vector
   * @param {Array<number>} vecB - Second vector
   * @returns {number} Cosine similarity score (0-1)
   */
  cosineSimilarity(vecA, vecB) {
    if (vecA.length !== vecB.length) {
      throw new Error('Vectors must have the same length');
    }

    const dotProduct = vecA.reduce((sum, a, i) => sum + a * vecB[i], 0);
    const magnitudeA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
    const magnitudeB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));

    if (magnitudeA === 0 || magnitudeB === 0) {
      return 0;
    }

    return dotProduct / (magnitudeA * magnitudeB);
  }

  /**
   * Create batches from an array
   * @param {Array} items - Items to batch
   * @param {number} batchSize - Size of each batch
   * @returns {Array<Array>} Array of batches
   */
  createBatches(items, batchSize) {
    const batches = [];
    for (let i = 0; i < items.length; i += batchSize) {
      batches.push(items.slice(i, i + batchSize));
    }
    return batches;
  }

  /**
   * Utility function to add delay
   * @param {number} ms - Milliseconds to delay
   */
  delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Get vector store statistics
   * @returns {Object} Statistics about the vector store
   */
  getStats() {
    const documents = [...new Set(this.embeddings.map(e => e.metadata.filename))];

    return {
      totalEmbeddings: this.embeddings.length,
      uniqueDocuments: documents.length,
      documents: documents,
      embeddingProvider: this.embeddingProvider.constructor.name,
      avgEmbeddingLength: this.embeddings.length > 0 ?
        this.embeddings[0].embedding.length : 0
    };
  }

  /**
   * Clear all embeddings from the vector store
   */
  clear() {
    this.embeddings = [];
    console.log('Vector store cleared');
  }

  /**
   * List all embeddings with content preview
   * @param {number} limit - Max number of embeddings to show
   * @returns {Array} Embeddings with content preview
   */
  listEmbeddings(limit = 10) {
    return this.embeddings.slice(0, limit).map(emb => ({
      id: emb.id,
      content: emb.content.substring(0, 100) + '...',
      filename: emb.metadata.filename,
      chunkIndex: emb.metadata.chunkIndex,
      embeddingSize: emb.embedding.length
    }));
  }

  /**
   * Get full embedding details by ID
   * @param {string} id - Embedding ID
   * @returns {Object} Full embedding object
   */
  getEmbedding(id) {
    return this.embeddings.find(emb => emb.id === id);
  }
}

module.exports = VectorStore;
