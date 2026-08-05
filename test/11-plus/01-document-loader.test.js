/**
 * Pipeline Step 1: DocumentLoader
 * Tests loadSingleDocument, cleanText, and chunkDocument
 * against the real 11-plus-syllabus.pdf
 */
const { test, describe, before } = require('node:test');
const assert = require('node:assert/strict');
const path = require('path');
const DocumentLoader = require('../../src/core/DocumentLoader');

const PDF_PATH = path.join(__dirname, '../../public/pdf-sources/11-plus-syllabus.pdf');
const EXPECTED_FILENAME = '11-plus-syllabus.pdf';

describe('DocumentLoader — 11-plus-syllabus.pdf', () => {
  let doc;

  before(async () => {
    doc = await DocumentLoader.loadSingleDocument(PDF_PATH);
  });

  // -------------------------------------------------------------------------
  // loadSingleDocument
  // -------------------------------------------------------------------------
  describe('loadSingleDocument', () => {
    test('returns a document object', () => {
      assert.ok(doc, 'document should exist');
      assert.ok(doc.content, 'document should have content');
      assert.ok(doc.metadata, 'document should have metadata');
    });

    test('metadata has correct filename', () => {
      assert.equal(doc.metadata.filename, EXPECTED_FILENAME);
    });

    test('metadata reports 6 pages', () => {
      assert.equal(doc.metadata.pages, 6);
    });

    test('content is a non-empty string', () => {
      assert.equal(typeof doc.content, 'string');
      assert.ok(doc.content.length > 0);
    });

    test('content is at least 10,000 characters', () => {
      assert.ok(doc.content.length >= 10000,
        `expected >= 10000 chars, got ${doc.content.length}`);
    });

    test('content contains known 11+ keywords', () => {
      assert.ok(doc.content.includes('ISEB'), 'should contain ISEB');
      assert.ok(doc.content.includes('Mathematics'), 'should contain Mathematics');
      assert.ok(doc.content.includes('English'), 'should contain English');
    });
  });

  // -------------------------------------------------------------------------
  // cleanText (tested via the loaded content)
  // -------------------------------------------------------------------------
  describe('cleanText', () => {
    test('no carriage returns in content', () => {
      assert.ok(!doc.content.includes('\r'),
        'carriage returns should have been stripped');
    });

    test('no control characters in content', () => {
      // eslint-disable-next-line no-control-regex
      const hasControl = /[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/.test(doc.content);
      assert.ok(!hasControl, 'control characters should have been stripped');
    });

    test('no runs of 3+ consecutive newlines', () => {
      assert.ok(!doc.content.includes('\n\n\n'),
        'excessive blank lines should be collapsed');
    });

    test('no runs of multiple spaces', () => {
      assert.ok(!/  /.test(doc.content),
        'multiple consecutive spaces should be collapsed');
    });

    test('content is trimmed', () => {
      assert.equal(doc.content[0], doc.content.trimStart()[0]);
      assert.equal(doc.content.at(-1), doc.content.trimEnd().at(-1));
    });

    // Direct unit test on cleanText with known dirty input
    test('cleanText strips control chars from raw input', () => {
      const dirty = "hello\x00world\x1Ftest";
      const clean = DocumentLoader.cleanText(dirty);
      assert.ok(!clean.includes('\x00'), 'null byte should be removed');
      assert.ok(!clean.includes('\x1F'), 'unit separator should be removed');
      // control chars are deleted (not replaced with space)
      assert.equal(clean, 'helloworldtest');
    });

    test('cleanText normalises windows line endings', () => {
      const result = DocumentLoader.cleanText("line1\r\nline2\r\nline3");
      assert.ok(!result.includes('\r'));
      assert.equal(result, "line1\nline2\nline3");
    });

    test('cleanText collapses multiple spaces', () => {
      const result = DocumentLoader.cleanText("too   many   spaces");
      assert.equal(result, "too many spaces");
    });
  });

  // -------------------------------------------------------------------------
  // chunkDocument
  // -------------------------------------------------------------------------
  describe('chunkDocument (default 1000/100)', () => {
    let chunks;

    before(async () => {
      chunks = await DocumentLoader.chunkDocument(doc.content, 1000, 100);
    });

    test('produces at least one chunk', () => {
      assert.ok(chunks.length > 0, 'should produce chunks');
    });

    test('produces expected number of chunks (~11-15 for this doc)', () => {
      assert.ok(chunks.length >= 10 && chunks.length <= 20,
        `expected 10-20 chunks, got ${chunks.length}`);
    });

    test('each chunk has required fields', () => {
      for (const chunk of chunks) {
        assert.ok('content' in chunk, 'chunk missing content');
        assert.ok('index' in chunk, 'chunk missing index');
        assert.ok('startChar' in chunk, 'chunk missing startChar');
        assert.ok('endChar' in chunk, 'chunk missing endChar');
        assert.ok('length' in chunk, 'chunk missing length');
      }
    });

    test('no chunk exceeds 1000 characters', () => {
      for (const chunk of chunks) {
        assert.ok(chunk.content.length <= 1000,
          `chunk ${chunk.index} is ${chunk.content.length} chars, exceeds 1000`);
      }
    });

    test('no chunk is empty', () => {
      for (const chunk of chunks) {
        assert.ok(chunk.content.trim().length > 0,
          `chunk ${chunk.index} is empty`);
      }
    });

    test('chunks are indexed sequentially from 0', () => {
      chunks.forEach((chunk, i) => {
        assert.equal(chunk.index, i, `chunk at position ${i} has index ${chunk.index}`);
      });
    });

    test('startChar of first chunk is 0', () => {
      assert.equal(chunks[0].startChar, 0);
    });

    test('consecutive chunks overlap — next startChar < prev endChar', () => {
      for (let i = 1; i < chunks.length; i++) {
        assert.ok(chunks[i].startChar < chunks[i - 1].endChar,
          `chunk ${i} does not overlap with chunk ${i - 1}`);
      }
    });

    test('no content is lost — all original text appears in some chunk', () => {
      // Sample 5 positions from the original doc and verify each appears in a chunk
      const samplePositions = [0, 2000, 4000, 7000, 10000].filter(p => p < doc.content.length);
      for (const pos of samplePositions) {
        const sample = doc.content.slice(pos, pos + 50).trim();
        if (sample.length < 10) continue; // skip if near end
        const found = chunks.some(c => c.content.includes(sample));
        assert.ok(found, `text at position ${pos} not found in any chunk: "${sample}"`);
      }
    });

    test('first chunk contains intro text about St Pauls', () => {
      assert.ok(chunks[0].content.includes("St Paul"),
        'first chunk should contain document opening');
    });

    test('later chunks contain maths topic content', () => {
      const mathsChunk = chunks.find(c =>
        c.content.includes('NUMBER') || c.content.includes('CALCULATIONS')
      );
      assert.ok(mathsChunk, 'should have a chunk with maths section content');
    });
  });

  // -------------------------------------------------------------------------
  // chunkDocument with smaller chunk size
  // -------------------------------------------------------------------------
  describe('chunkDocument (small 500/50)', () => {
    let smallChunks;

    before(async () => {
      smallChunks = await DocumentLoader.chunkDocument(doc.content, 500, 50);
    });

    test('produces more chunks than default size', async () => {
      const defaultChunks = await DocumentLoader.chunkDocument(doc.content, 1000, 100);
      assert.ok(smallChunks.length > defaultChunks.length,
        'smaller chunk size should produce more chunks');
    });

    test('no chunk exceeds 500 characters', () => {
      for (const chunk of smallChunks) {
        assert.ok(chunk.content.length <= 500,
          `chunk ${chunk.index} exceeds 500 chars`);
      }
    });
  });
});
