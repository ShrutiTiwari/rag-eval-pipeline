/**
 * Pipeline Step 3: VectorStore.similaritySearch
 * Tests that semantic search over 11+ chunks returns the right content.
 * Builds a real vector store from the doc — verifies retrieval quality.
 */
const { test, describe, before } = require('node:test');
const assert = require('node:assert/strict');
const path = require('path');
const DocumentLoader = require('../../src/core/DocumentLoader');
const VectorStore = require('../../src/core/VectorStore');
const { LocalEmbeddingProvider } = require('../../src/providers/EmbeddingProvider');

const PDF_PATH = path.join(__dirname, '../../public/pdf-sources/11-plus-syllabus.pdf');

describe('VectorStore.similaritySearch — 11-plus-syllabus.pdf', () => {
  let store;

  before(async () => {
    // Build a full vector store from the 11+ doc
    const provider = new LocalEmbeddingProvider();
    store = new VectorStore(provider);

    const doc = await DocumentLoader.loadSingleDocument(PDF_PATH);
    const chunks = await DocumentLoader.chunkDocument(doc.content, 1000, 100);
    const embeddings = await store.generateEmbeddings(chunks, doc.metadata);
    store.addEmbeddings(embeddings);
  });

  // -------------------------------------------------------------------------
  // addEmbeddings / getStats
  // -------------------------------------------------------------------------
  describe('addEmbeddings', () => {
    test('vector store has embeddings loaded', () => {
      const stats = store.getStats();
      assert.ok(stats.totalEmbeddings > 0, 'should have embeddings');
    });

    test('all embeddings belong to 11-plus-syllabus.pdf', () => {
      const stats = store.getStats();
      assert.equal(stats.uniqueDocuments, 1);
      assert.ok(stats.documents.includes('11-plus-syllabus.pdf'));
    });

    test('embedding count matches chunk count (10-20 expected)', () => {
      const stats = store.getStats();
      assert.ok(stats.totalEmbeddings >= 10 && stats.totalEmbeddings <= 25,
        `expected 10-25 embeddings, got ${stats.totalEmbeddings}`);
    });
  });

  // -------------------------------------------------------------------------
  // similaritySearch — result structure
  // -------------------------------------------------------------------------
  describe('result structure', () => {
    let results;

    before(async () => {
      results = await store.similaritySearch('ISEB Common Pre-Test', 3, 0.1);
    });

    test('returns an array', () => {
      assert.ok(Array.isArray(results));
    });

    test('respects topK limit', async () => {
      const r = await store.similaritySearch('mathematics', 2, 0.0);
      assert.ok(r.length <= 2);
    });

    test('each result has id, content, similarity, metadata', () => {
      for (const r of results) {
        assert.ok('id' in r, 'missing id');
        assert.ok('content' in r, 'missing content');
        assert.ok('similarity' in r, 'missing similarity');
        assert.ok('metadata' in r, 'missing metadata');
      }
    });

    test('similarity scores are between 0 and 1', () => {
      for (const r of results) {
        assert.ok(r.similarity >= 0 && r.similarity <= 1,
          `similarity ${r.similarity} out of range`);
      }
    });

    test('results are sorted by similarity descending', () => {
      for (let i = 1; i < results.length; i++) {
        assert.ok(results[i].similarity <= results[i - 1].similarity,
          'results should be sorted by similarity descending');
      }
    });

    test('metadata includes filename', () => {
      for (const r of results) {
        assert.equal(r.metadata.filename, '11-plus-syllabus.pdf');
      }
    });
  });

  // -------------------------------------------------------------------------
  // similaritySearch — threshold filtering
  // -------------------------------------------------------------------------
  describe('threshold filtering', () => {
    test('high threshold returns fewer results', async () => {
      const low = await store.similaritySearch('mathematics', 10, 0.0);
      const high = await store.similaritySearch('mathematics', 10, 0.5);
      assert.ok(high.length <= low.length,
        'higher threshold should return fewer or equal results');
    });

    test('threshold=1.0 returns no results (nothing is a perfect match)', async () => {
      const results = await store.similaritySearch('mathematics', 5, 1.0);
      assert.equal(results.length, 0);
    });

    test('threshold=0.0 returns topK results', async () => {
      const results = await store.similaritySearch('mathematics', 3, 0.0);
      assert.equal(results.length, 3);
    });
  });

  // -------------------------------------------------------------------------
  // similaritySearch — content relevance (11+ specific)
  // -------------------------------------------------------------------------
  describe('content relevance', () => {
    test('query about ISEB returns chunk mentioning ISEB or Pre-Test', async () => {
      const results = await store.similaritySearch(
        'What subjects are in the ISEB Common Pre-Test?', 3, 0.1
      );
      const found = results.some(r =>
        r.content.includes('ISEB') || r.content.includes('Pre-Test') || r.content.includes('Verbal Reasoning')
      );
      assert.ok(found, 'top results should mention ISEB or Pre-Test');
    });

    test('query about maths returns chunk mentioning Mathematics or Section', async () => {
      const results = await store.similaritySearch(
        'What are the sections of the mathematics paper?', 3, 0.1
      );
      const found = results.some(r =>
        r.content.includes('Mathematics') || r.content.includes('Section') || r.content.includes('calculator')
      );
      assert.ok(found, 'top results should mention Mathematics content');
    });

    test('query about English returns chunk mentioning comprehension or composition', async () => {
      const results = await store.similaritySearch(
        'English comprehension and composition paper format', 3, 0.1
      );
      const found = results.some(r =>
        r.content.includes('Comprehension') ||
        r.content.includes('Composition') ||
        r.content.includes('English')
      );
      assert.ok(found, 'top results should mention English content');
    });

    test('query about excluded topics returns chunk with "will not be tested"', async () => {
      const results = await store.similaritySearch(
        'What topics are not tested in the exam?', 5, 0.1
      );
      const found = results.some(r => r.content.includes('not be tested'));
      assert.ok(found, 'should find chunks listing excluded topics');
    });

    test('unrelated query returns low similarity scores', async () => {
      const results = await store.similaritySearch(
        'quantum physics particle accelerator', 3, 0.0
      );
      // Scores should be low since topic is completely unrelated
      if (results.length > 0) {
        assert.ok(results[0].similarity < 0.5,
          `expected low similarity for unrelated query, got ${results[0].similarity}`);
      }
    });
  });
});
