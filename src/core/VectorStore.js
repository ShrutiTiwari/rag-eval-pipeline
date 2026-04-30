const OpenAI = require('openai');

/**
 * VectorStore - Handles document embeddings and similarity search
 * for the RAG chatbot system using OpenAI embeddings
 */
class VectorStore {
  constructor(apiKey = null) {
    this.openai = new OpenAI({
      apiKey: apiKey || process.env.OPENAI_API_KEY
    });
    this.embeddings = []; // Store embeddings with metadata
    this.embeddingModel = 'text-embedding-3-small'; // Cost-effective embedding model
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

    console.log(`🔮 Generating embeddings for ${chunks.length} chunks using ${this.embeddingModel}...`);
    const embeddings = [];

    // Process chunks in batches to avoid rate limits
    const batchSize = 100; // OpenAI allows up to 2048 inputs per request
    const batches = this.createBatches(chunks, batchSize);

    for (let i = 0; i < batches.length; i++) {
      const batch = batches[i];
      console.log(`📦 Processing batch ${i + 1}/${batches.length} (${batch.length} chunks)`);

      try {
        const response = await this.openai.embeddings.create({
          model: this.embeddingModel,
          input: batch.map(chunk => chunk.content)
        });

        // Combine embeddings with chunk metadata
        response.data.forEach((embeddingData, index) => {
          const chunk = batch[index];
          embeddings.push({
            id: `${metadata.filename || 'doc'}_chunk_${chunk.index || index}`,
            content: chunk.content,
            embedding: embeddingData.embedding,
            metadata: {
              ...metadata,
              chunkIndex: chunk.index || index,
              startChar: chunk.startChar,
              endChar: chunk.endChar,
              length: chunk.length || chunk.content.length
            },
            createdAt: new Date().toISOString()
          });
        });

        // Add delay between batches to respect rate limits
        if (i < batches.length - 1) {
          await this.delay(100); // 100ms delay between batches
        }
      } catch (error) {
        console.error(`❌ Error processing batch ${i + 1}:`, error.message);
        throw new Error(`Failed to generate embeddings for batch ${i + 1}: ${error.message}`);
      }
    }

    console.log(`✅ Generated ${embeddings.length} embeddings successfully`);
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
    console.log(`📚 Added ${embeddings.length} embeddings to vector store (Total: ${this.embeddings.length})`);
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

    try {
      const response = await this.openai.embeddings.create({
        model: this.embeddingModel,
        input: query.trim()
      });

      return response.data[0].embedding;
    } catch (error) {
      console.error('❌ Error generating query embedding:', error.message);
      throw new Error(`Failed to generate query embedding: ${error.message}`);
    }
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

    console.log(`🔍 Searching for: "${query}" (topK=${topK}, threshold=${threshold})`);

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

    console.log(`📊 Found ${filteredResults.length} results above threshold ${threshold}`);

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
      model: this.embeddingModel,
      avgEmbeddingLength: this.embeddings.length > 0 ?
        this.embeddings[0].embedding.length : 0
    };
  }

  /**
   * Clear all embeddings from the vector store
   */
  clear() {
    this.embeddings = [];
    console.log('🗑️  Vector store cleared');
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