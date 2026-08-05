/**
 * Pipeline Step 2: EmbeddingProvider + VectorStore.generateEmbeddings
 * Tests that chunks from the 11+ doc are embedded correctly.
 * Uses the local model — no API key needed, but downloads ~25MB on first run.
 */
const { test, describe, before } = require('node:test');
const assert = require('node:assert/strict');
const path = require('path');
const DocumentLoader = require('../../src/core/DocumentLoader');
const VectorStore = require('../../src/core/VectorStore');
const { LocalEmbeddingProvider } = require('../../src/providers/EmbeddingProvider');

const PDF_PATH = path.join(__dirname, '../../public/pdf-sources/11-plus-syllabus.pdf');
const MINILM_DIMENSIONS = 384; // all-MiniLM-L6-v2 output size

describe('EmbeddingProvider — 11-plus-syllabus.pdf', () => {
  let provider;
  let chunks;

  before(async () => {
    provider = new LocalEmbeddingProvider();
    const doc = await DocumentLoader.loadSingleDocument(PDF_PATH);
    chunks = await DocumentLoader.chunkDocument(doc.content, 1000, 100);
  });

  // -------------------------------------------------------------------------
  // LocalEmbeddingProvider.embed
  // -------------------------------------------------------------------------
  describe('LocalEmbeddingProvider.embed', () => {
    test('returns one vector per input text', async () => {
      const texts = [chunks[0].content, chunks[1].content];
      const vectors = await provider.embed(texts);
      assert.equal(vectors.length, 2);
    });

    test('each vector has 384 dimensions (MiniLM)', async () => {
      const vectors = await provider.embed([chunks[0].content]);
      assert.equal(vectors[0].length, MINILM_DIMENSIONS,
        `expected ${MINILM_DIMENSIONS} dims, got ${vectors[0].length}`);
    });

    test('all vector values are finite numbers', async () => {
      const vectors = await provider.embed([chunks[0].content]);
      for (const val of vectors[0]) {
        assert.equal(typeof val, 'number');
        assert.ok(isFinite(val), `non-finite value in vector: ${val}`);
      }
    });

    test('different texts produce different vectors', async () => {
      const vectors = await provider.embed([chunks[0].content, chunks[5].content]);
      const identical = vectors[0].every((v, i) => v === vectors[1][i]);
      assert.ok(!identical, 'different chunks should produce different vectors');
    });

    test('same text produces identical vectors (deterministic)', async () => {
      const text = chunks[0].content;
      const [v1] = await provider.embed([text]);
      const [v2] = await provider.embed([text]);
      assert.deepEqual(v1, v2, 'same input should give same embedding');
    });

    test('vectors are normalised — magnitude ≈ 1.0', async () => {
      const [vec] = await provider.embed([chunks[0].content]);
      const magnitude = Math.sqrt(vec.reduce((sum, v) => sum + v * v, 0));
      assert.ok(Math.abs(magnitude - 1.0) < 0.001,
        `vector magnitude should be ~1.0, got ${magnitude}`);
    });
  });

  // -------------------------------------------------------------------------
  // LocalEmbeddingProvider.embedOne
  // -------------------------------------------------------------------------
  describe('LocalEmbeddingProvider.embedOne', () => {
    test('returns a single flat vector', async () => {
      const vec = await provider.embedOne(chunks[0].content);
      assert.ok(Array.isArray(vec));
      assert.equal(vec.length, MINILM_DIMENSIONS);
    });

    test('result matches embed([text])[0]', async () => {
      const text = chunks[2].content;
      const fromOne = await provider.embedOne(text);
      const [fromBatch] = await provider.embed([text]);
      assert.deepEqual(fromOne, fromBatch);
    });
  });

  // -------------------------------------------------------------------------
  // VectorStore.generateEmbeddings
  // -------------------------------------------------------------------------
  describe('VectorStore.generateEmbeddings', () => {
    let store;
    let embeddings;

    before(async () => {
      store = new VectorStore(provider);
      // Use first 5 chunks to keep test fast
      embeddings = await store.generateEmbeddings(chunks.slice(0, 5), {
        filename: '11-plus-syllabus.pdf',
        pages: 6
      });
    });

    test('returns one embedding object per chunk', () => {
      assert.equal(embeddings.length, 5);
    });

    test('each embedding has required fields', () => {
      for (const emb of embeddings) {
        assert.ok('id' in emb, 'missing id');
        assert.ok('content' in emb, 'missing content');
        assert.ok('embedding' in emb, 'missing embedding');
        assert.ok('metadata' in emb, 'missing metadata');
        assert.ok('createdAt' in emb, 'missing createdAt');
      }
    });

    test('embedding id includes filename and chunk index', () => {
      assert.ok(embeddings[0].id.includes('11-plus-syllabus.pdf'));
      assert.ok(embeddings[0].id.includes('chunk_0'));
    });

    test('content matches original chunk content', () => {
      for (let i = 0; i < 5; i++) {
        assert.equal(embeddings[i].content, chunks[i].content);
      }
    });

    test('metadata includes filename from doc metadata', () => {
      assert.equal(embeddings[0].metadata.filename, '11-plus-syllabus.pdf');
    });

    test('metadata includes chunkIndex', () => {
      for (let i = 0; i < 5; i++) {
        assert.equal(embeddings[i].metadata.chunkIndex, i);
      }
    });

    test('metadata includes startChar and endChar', () => {
      for (const emb of embeddings) {
        assert.ok('startChar' in emb.metadata);
        assert.ok('endChar' in emb.metadata);
        assert.ok(emb.metadata.endChar > emb.metadata.startChar);
      }
    });

    test('embedding vector has correct dimensions', () => {
      for (const emb of embeddings) {
        assert.equal(emb.embedding.length, MINILM_DIMENSIONS);
      }
    });
  });
});
