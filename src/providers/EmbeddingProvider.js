/**
 * EmbeddingProvider - Base class and concrete implementations for generating text embeddings.
 * Swap the embedding model by setting EMBEDDING_PROVIDER=openai (or 'local') in your .env.
 */

// ---------------------------------------------------------------------------
// Base class
// ---------------------------------------------------------------------------
class EmbeddingProvider {
  /**
   * Generate embeddings for a batch of texts.
   * @param {string[]} texts
   * @returns {Promise<number[][]>}
   */
  async embed(texts) {
    throw new Error('embed() must be implemented by a subclass');
  }

  /**
   * Generate embedding for a single text.
   * @param {string} text
   * @returns {Promise<number[]>}
   */
  async embedOne(text) {
    const results = await this.embed([text]);
    return results[0];
  }
}

// ---------------------------------------------------------------------------
// Local (Xenova/transformers) implementation
// ---------------------------------------------------------------------------
class LocalEmbeddingProvider extends EmbeddingProvider {
  constructor(model = 'Xenova/all-MiniLM-L6-v2') {
    super();
    this.model = model;
    this._pipeline = null; // lazy-loaded on first call
  }

  async _getPipeline() {
    if (!this._pipeline) {
      console.log('Loading local embedding model (first time only, downloads ~25MB)...');
      const { pipeline } = await import('@xenova/transformers');
      this._pipeline = await pipeline('feature-extraction', this.model);
      console.log('Local embedding model ready');
    }
    return this._pipeline;
  }

  async embed(texts) {
    if (!Array.isArray(texts) || texts.length === 0) {
      throw new Error('embed() requires a non-empty array of strings');
    }

    try {
      const pipe = await this._getPipeline();
      const embeddings = [];

      for (const text of texts) {
        const output = await pipe(text, { pooling: 'mean', normalize: true });
        embeddings.push(Array.from(output.data));
      }

      return embeddings;
    } catch (error) {
      throw new Error(`LocalEmbeddingProvider.embed() failed: ${error.message}`);
    }
  }
}

// ---------------------------------------------------------------------------
// OpenAI implementation
// ---------------------------------------------------------------------------
class OpenAIEmbeddingProvider extends EmbeddingProvider {
  constructor(model = 'text-embedding-3-small') {
    super();
    this.model = model;
    this.apiKey = process.env.OPENAI_API_KEY;

    if (!this.apiKey) {
      throw new Error('OpenAIEmbeddingProvider requires OPENAI_API_KEY to be set in the environment');
    }

    const OpenAI = require('openai');
    this.client = new OpenAI({ apiKey: this.apiKey });
  }

  async embed(texts) {
    if (!Array.isArray(texts) || texts.length === 0) {
      throw new Error('embed() requires a non-empty array of strings');
    }

    try {
      const response = await this.client.embeddings.create({
        model: this.model,
        input: texts
      });

      // OpenAI returns embeddings in the same order as the input
      return response.data.map(item => item.embedding);
    } catch (error) {
      throw new Error(`OpenAIEmbeddingProvider.embed() failed: ${error.message}`);
    }
  }
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------
function createEmbeddingProvider(type = process.env.EMBEDDING_PROVIDER || 'local') {
  if (type === 'openai') return new OpenAIEmbeddingProvider();
  return new LocalEmbeddingProvider();
}

module.exports = { createEmbeddingProvider, LocalEmbeddingProvider, OpenAIEmbeddingProvider };
