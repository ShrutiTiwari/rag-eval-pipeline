/**
 * Pipeline Step 1b: ChunkingStrategy
 * Tests FixedSizeChunking and SectionAwareChunking against real 11-plus content.
 * Also compares retrieval quality between strategies for a known tricky query.
 */
const { test, describe, before } = require('node:test');
const assert = require('node:assert/strict');
const path = require('path');

const { FixedSizeChunking, SectionAwareChunking, createChunkingStrategy } = require('../../src/providers/ChunkingStrategy');
const DocumentLoader = require('../../src/core/DocumentLoader');
const VectorStore = require('../../src/core/VectorStore');

const PDF_PATH = path.join(__dirname, '../../public/pdf-sources/11-plus-syllabus.pdf');

// ---------------------------------------------------------------------------
// FixedSizeChunking
// ---------------------------------------------------------------------------
describe('FixedSizeChunking', () => {
  const CHUNK_SIZE = 1000;
  const OVERLAP = 100;
  let content;
  let chunks;

  before(async () => {
    const doc = await DocumentLoader.loadSingleDocument(PDF_PATH);
    content = doc.content;
    const strategy = new FixedSizeChunking(CHUNK_SIZE, OVERLAP);
    chunks = strategy.chunk(content);
  });

  test('returns a non-empty array', () => {
    assert.ok(Array.isArray(chunks), 'chunks should be an array');
    assert.ok(chunks.length > 0, 'should have at least one chunk');
  });

  test('produces multiple chunks for a multi-page document', () => {
    assert.ok(chunks.length >= 5,
      `expected >= 5 chunks, got ${chunks.length}`);
  });

  test('each chunk has required shape fields', () => {
    for (const chunk of chunks) {
      assert.ok(typeof chunk.content === 'string', 'content must be string');
      assert.ok(typeof chunk.index === 'number', 'index must be number');
      assert.ok(typeof chunk.startChar === 'number', 'startChar must be number');
      assert.ok(typeof chunk.endChar === 'number', 'endChar must be number');
      assert.ok(typeof chunk.length === 'number', 'length must be number');
      assert.equal(chunk.strategy, 'FixedSizeChunking', 'strategy name must match class');
    }
  });

  test('chunk indices are sequential starting from 0', () => {
    for (let i = 0; i < chunks.length; i++) {
      assert.equal(chunks[i].index, i, `chunk ${i} should have index ${i}`);
    }
  });

  test('no chunk exceeds chunkSize characters', () => {
    for (const chunk of chunks) {
      assert.ok(chunk.content.length <= CHUNK_SIZE,
        `chunk ${chunk.index} length ${chunk.content.length} exceeds ${CHUNK_SIZE}`);
    }
  });

  test('no chunk is empty', () => {
    for (const chunk of chunks) {
      assert.ok(chunk.content.trim().length > 0,
        `chunk ${chunk.index} is empty`);
    }
  });

  test('startChar of first chunk is 0', () => {
    assert.equal(chunks[0].startChar, 0);
  });

  test('startChar increases monotonically', () => {
    for (let i = 1; i < chunks.length; i++) {
      assert.ok(chunks[i].startChar > chunks[i - 1].startChar,
        `chunk ${i} startChar ${chunks[i].startChar} should be > chunk ${i-1} startChar ${chunks[i-1].startChar}`);
    }
  });

  test('endChar > startChar for every chunk', () => {
    for (const chunk of chunks) {
      assert.ok(chunk.endChar > chunk.startChar,
        `chunk ${chunk.index}: endChar ${chunk.endChar} should be > startChar ${chunk.startChar}`);
    }
  });

  test('content contains known 11+ keywords somewhere across all chunks', () => {
    const allText = chunks.map(c => c.content).join(' ');
    assert.ok(allText.includes('Mathematics'), 'should contain Mathematics');
    assert.ok(allText.includes('English'), 'should contain English');
  });

  test('chunk(empty string) returns empty array', () => {
    const strategy = new FixedSizeChunking();
    assert.deepEqual(strategy.chunk(''), []);
  });

  test('chunk(null) returns empty array', () => {
    const strategy = new FixedSizeChunking();
    assert.deepEqual(strategy.chunk(null), []);
  });

  test('single short text produces exactly one chunk', () => {
    const strategy = new FixedSizeChunking(1000, 100);
    const result = strategy.chunk('Hello world.');
    assert.equal(result.length, 1);
    assert.equal(result[0].content, 'Hello world.');
    assert.equal(result[0].index, 0);
  });

  test('custom chunk size is respected', () => {
    const strategy = new FixedSizeChunking(200, 20);
    const longText = 'a'.repeat(1000);
    const result = strategy.chunk(longText);
    assert.ok(result.length > 1, 'should split into multiple chunks');
    for (const chunk of result) {
      assert.ok(chunk.content.length <= 200, `chunk too long: ${chunk.content.length}`);
    }
  });
});

// ---------------------------------------------------------------------------
// SectionAwareChunking
// ---------------------------------------------------------------------------
describe('SectionAwareChunking', () => {
  let content;
  let chunks;

  before(async () => {
    const doc = await DocumentLoader.loadSingleDocument(PDF_PATH);
    content = doc.content;
    const strategy = new SectionAwareChunking(1500, 150);
    chunks = strategy.chunk(content);
  });

  test('returns a non-empty array', () => {
    assert.ok(Array.isArray(chunks), 'chunks should be an array');
    assert.ok(chunks.length > 0, 'should have at least one chunk');
  });

  test('each chunk has required shape fields', () => {
    for (const chunk of chunks) {
      assert.ok(typeof chunk.content === 'string', 'content must be string');
      assert.ok(typeof chunk.index === 'number', 'index must be number');
      assert.ok(typeof chunk.startChar === 'number', 'startChar must be number');
      assert.ok(typeof chunk.endChar === 'number', 'endChar must be number');
      assert.ok(typeof chunk.length === 'number', 'length must be number');
      assert.equal(chunk.strategy, 'SectionAwareChunking');
    }
  });

  test('chunk indices are sequential starting from 0', () => {
    for (let i = 0; i < chunks.length; i++) {
      assert.equal(chunks[i].index, i);
    }
  });

  test('each chunk has a section label', () => {
    for (const chunk of chunks) {
      assert.ok(typeof chunk.section === 'string' && chunk.section.length > 0,
        `chunk ${chunk.index} missing section label`);
    }
  });

  test('no chunk is empty', () => {
    for (const chunk of chunks) {
      assert.ok(chunk.content.trim().length > 0,
        `chunk ${chunk.index} (section: ${chunk.section}) is empty`);
    }
  });

  test('Mathematics and English content are in separate chunks', () => {
    // Find chunks that contain substantial Mathematics and English content
    const mathChunks = chunks.filter(c =>
      c.content.toLowerCase().includes('mathematics') ||
      c.content.toLowerCase().includes('calculator') ||
      (c.section && c.section.toLowerCase().includes('mathematics'))
    );
    const englishChunks = chunks.filter(c =>
      c.content.toLowerCase().includes('comprehension') ||
      c.content.toLowerCase().includes('composition') ||
      (c.section && c.section.toLowerCase().includes('english'))
    );

    assert.ok(mathChunks.length > 0, 'should have chunks with mathematics content');
    assert.ok(englishChunks.length > 0, 'should have chunks with english content');

    // No chunk should mix both calculator content and comprehension content
    const mixedChunks = chunks.filter(c =>
      (c.content.toLowerCase().includes('calculator') ||
       c.content.toLowerCase().includes('arithmetic')) &&
      c.content.toLowerCase().includes('comprehension')
    );
    assert.equal(mixedChunks.length, 0,
      `Found ${mixedChunks.length} chunk(s) mixing maths and english content — sections are bleeding`);
  });

  test('no chunk exceeds maxSectionSize * 1.5 characters (sub-splitting working)', () => {
    const MAX = 1500 * 1.5;
    for (const chunk of chunks) {
      assert.ok(chunk.content.length <= MAX,
        `chunk ${chunk.index} length ${chunk.content.length} exceeds max`);
    }
  });

  test('chunk(empty string) returns empty array', () => {
    const strategy = new SectionAwareChunking();
    assert.deepEqual(strategy.chunk(''), []);
  });

  test('chunk(null) returns empty array', () => {
    const strategy = new SectionAwareChunking();
    assert.deepEqual(strategy.chunk(null), []);
  });

  test('recognises ALL CAPS header as section boundary', () => {
    const strategy = new SectionAwareChunking();
    const text = 'Introduction text here.\n\nMATHEMATICS\nThis is the maths section.\n\nENGLISH\nThis is english.';
    const result = strategy.chunk(text);
    assert.ok(result.length >= 2, `expected >= 2 sections, got ${result.length}`);
    const mathChunk = result.find(c => c.section === 'MATHEMATICS');
    assert.ok(mathChunk, 'should have a MATHEMATICS section chunk');
    assert.ok(mathChunk.content.includes('maths section'));
  });

  test('recognises "Part A:" style header as section boundary', () => {
    const strategy = new SectionAwareChunking();
    const text = 'Preamble text.\nPart A: Reading\nContent for part A.\nPart B: Writing\nContent for part B.';
    const result = strategy.chunk(text);
    const partAChunk = result.find(c => c.section && c.section.startsWith('Part A'));
    assert.ok(partAChunk, 'should detect Part A: as a header');
  });
});

// ---------------------------------------------------------------------------
// Factory: createChunkingStrategy
// ---------------------------------------------------------------------------
describe('createChunkingStrategy factory', () => {
  test('returns FixedSizeChunking by default', () => {
    const strategy = createChunkingStrategy('fixed');
    assert.ok(strategy instanceof FixedSizeChunking);
  });

  test('returns SectionAwareChunking when type=section', () => {
    const strategy = createChunkingStrategy('section');
    assert.ok(strategy instanceof SectionAwareChunking);
  });

  test('respects custom chunkSize option for fixed strategy', () => {
    const strategy = createChunkingStrategy('fixed', { chunkSize: 500, overlap: 50 });
    assert.equal(strategy.chunkSize, 500);
    assert.equal(strategy.overlap, 50);
  });

  test('respects custom maxSectionSize option for section strategy', () => {
    const strategy = createChunkingStrategy('section', { maxSectionSize: 2000, overlap: 200 });
    assert.equal(strategy.maxSectionSize, 2000);
    assert.equal(strategy.overlap, 200);
  });

  test('unknown type falls back to FixedSizeChunking', () => {
    const strategy = createChunkingStrategy('unknown');
    assert.ok(strategy instanceof FixedSizeChunking);
  });
});

// ---------------------------------------------------------------------------
// Retrieval quality comparison: SectionAware vs Fixed
// ---------------------------------------------------------------------------
describe('Retrieval quality — SectionAware vs Fixed for maths query', () => {
  const QUERY = 'How long is the mathematics paper and is a calculator allowed?';
  let fixedResults;
  let sectionResults;

  before(async () => {
    const doc = await DocumentLoader.loadSingleDocument(PDF_PATH);

    // Fixed strategy store
    const fixedStrategy = new FixedSizeChunking(1000, 100);
    const fixedChunks = fixedStrategy.chunk(doc.content);
    const fixedStore = new VectorStore();
    const fixedEmbeddings = await fixedStore.generateEmbeddings(fixedChunks, doc.metadata);
    fixedStore.addEmbeddings(fixedEmbeddings);
    fixedResults = await fixedStore.similaritySearch(QUERY, 3, 0.0);

    // Section strategy store
    const sectionStrategy = new SectionAwareChunking(1500, 150);
    const sectionChunks = sectionStrategy.chunk(doc.content);
    const sectionStore = new VectorStore();
    const sectionEmbeddings = await sectionStore.generateEmbeddings(sectionChunks, doc.metadata);
    sectionStore.addEmbeddings(sectionEmbeddings);
    sectionResults = await sectionStore.similaritySearch(QUERY, 3, 0.0);
  });

  test('both strategies return results', () => {
    assert.ok(fixedResults.length > 0, 'fixed strategy should return results');
    assert.ok(sectionResults.length > 0, 'section strategy should return results');
  });

  test('section strategy top result contains calculator-related content', () => {
    const top = sectionResults[0];
    const lc = top.content.toLowerCase();
    assert.ok(
      lc.includes('calculator') || lc.includes('arithmetic') || lc.includes('minutes'),
      `Top section chunk should mention calculator/time. Got: "${top.content.substring(0, 200)}"`
    );
  });

  test('section strategy top similarity is at least 0.5', () => {
    assert.ok(sectionResults[0].similarity >= 0.5,
      `Expected >= 0.5, got ${sectionResults[0].similarity.toFixed(3)}`);
  });

  test('section strategy top result has section metadata', () => {
    const top = sectionResults[0];
    assert.ok(top.metadata.section, 'section metadata should be present');
    assert.equal(top.metadata.strategy, 'SectionAwareChunking');
  });

  test('section strategy top similarity >= fixed strategy top similarity', () => {
    const sectionTop = sectionResults[0].similarity;
    const fixedTop = fixedResults[0].similarity;
    assert.ok(sectionTop >= fixedTop,
      `Section (${sectionTop.toFixed(3)}) should be >= Fixed (${fixedTop.toFixed(3)}) for this query`);
  });
});
