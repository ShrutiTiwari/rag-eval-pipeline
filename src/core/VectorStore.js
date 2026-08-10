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
