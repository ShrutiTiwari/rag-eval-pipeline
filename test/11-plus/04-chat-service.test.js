/**
 * Pipeline Step 4: ChatService
 * Tests that ChatService initializes correctly, builds proper messages,
 * assembles context, and runs the full RAG chat loop.
 * All tests except the final smoke test use a StubLLMProvider to avoid API costs.
 */
require('dotenv').config();

const { test, describe, before } = require('node:test');
const assert = require('node:assert/strict');
const path = require('path');

const ChatService = require('../../src/core/ChatService');
const DocumentLoader = require('../../src/core/DocumentLoader');
const { LocalEmbeddingProvider } = require('../../src/providers/EmbeddingProvider');
const { ClaudeLLMProvider } = require('../../src/providers/LLMProvider');
const VectorStore = require('../../src/core/VectorStore');

const PDF_PATH = path.join(__dirname, '../../public/pdf-sources/11-plus-syllabus.pdf');

// ---------------------------------------------------------------------------
// Stub LLM provider — avoids real API calls for unit-style tests
// ---------------------------------------------------------------------------
class StubLLMProvider {
  constructor(reply = 'Stub reply') {
    this.reply = reply;
    this.lastMessages = null;
  }

  async chat(messages, options) {
    this.lastMessages = messages;
    return this.reply;
  }
}

// ---------------------------------------------------------------------------
// Helper — build a minimal document object that ChatService.initializeWithDocuments accepts
// ---------------------------------------------------------------------------
async function loadDoc() {
  return DocumentLoader.loadSingleDocument(PDF_PATH);
}

// ---------------------------------------------------------------------------
// initializeWithDocuments
// ---------------------------------------------------------------------------
describe('ChatService.initializeWithDocuments', () => {
  let service;
  let doc;

  before(async () => {
    doc = await loadDoc();
    service = new ChatService(new StubLLMProvider());
    await service.initializeWithDocuments([doc]);
  });

  test('vectorStore is populated after init', () => {
    assert.ok(service.vectorStore !== null, 'vectorStore should not be null');
    const stats = service.vectorStore.getStats();
    assert.ok(stats.totalEmbeddings > 0, `expected embeddings, got ${stats.totalEmbeddings}`);
  });

  test('documents array has 1 entry with correct filename', () => {
    assert.equal(service.documents.length, 1);
    assert.equal(service.documents[0].metadata.filename, '11-plus-syllabus.pdf');
  });

  test('evaluator is initialized (not null)', () => {
    assert.ok(service.evaluator !== null, 'evaluator should be set');
  });

  test('throws if called with empty array', async () => {
    const s = new ChatService(new StubLLMProvider());
    await assert.rejects(
      () => s.initializeWithDocuments([]),
      /No documents provided/
    );
  });
});

// ---------------------------------------------------------------------------
// assembleContext
// ---------------------------------------------------------------------------
describe('ChatService.assembleContext', () => {
  let service;

  before(() => {
    service = new ChatService(new StubLLMProvider());
    // No init needed — assembleContext is pure
  });

  test('formats retrieved results with [Source N: filename, chunk X] header', () => {
    const fakeResult = {
      content: 'Some content here',
      metadata: { filename: 'test.pdf', chunkIndex: 3 }
    };
    const ctx = service.assembleContext([fakeResult]);
    assert.ok(ctx.includes('[Source 1: test.pdf, chunk 3]'), `header missing; got: ${ctx}`);
  });

  test('multiple results are separated by ---', () => {
    const results = [
      { content: 'Content A', metadata: { filename: 'a.pdf', chunkIndex: 0 } },
      { content: 'Content B', metadata: { filename: 'b.pdf', chunkIndex: 1 } }
    ];
    const ctx = service.assembleContext(results);
    assert.ok(ctx.includes('---'), 'separator --- should appear between results');
  });

  test('long context is truncated when exceeding maxContextLength * 4 chars', () => {
    // Each result needs to be large enough that the combined context exceeds maxContextLength * 4.
    // maxContextLength defaults to 8000, so the limit is 32000 chars.
    // Use 12000 chars per chunk — 3 results give ~36000 chars which exceeds the limit.
    const longContent = 'x'.repeat(12000);
    const results = [
      { content: longContent, metadata: { filename: 'a.pdf', chunkIndex: 0 } },
      { content: longContent, metadata: { filename: 'b.pdf', chunkIndex: 1 } },
      { content: longContent, metadata: { filename: 'c.pdf', chunkIndex: 2 } }
    ];
    const ctx = service.assembleContext(results);
    const maxLen = service.maxContextLength * 4;
    // The output should be the truncated prefix + the truncation notice
    assert.ok(
      ctx.includes('[Context truncated...]'),
      'truncation notice should appear'
    );
    assert.ok(
      ctx.length <= maxLen + 100, // small buffer for the trailing notice
      `context length ${ctx.length} exceeds limit`
    );
  });
});

// ---------------------------------------------------------------------------
// buildMessages
// ---------------------------------------------------------------------------
describe('ChatService.buildMessages', () => {
  let service;
  const ctx = 'Fake context paragraph.';
  const query = 'What is tested?';

  before(() => {
    service = new ChatService(new StubLLMProvider());
  });

  test('first message role is system', () => {
    const msgs = service.buildMessages(query, ctx, false);
    assert.equal(msgs[0].role, 'system');
  });

  test('system message contains the context string', () => {
    const msgs = service.buildMessages(query, ctx, false);
    assert.ok(msgs[0].content.includes(ctx), 'system message should embed the context');
  });

  test('last message role is user and content matches the user query', () => {
    const msgs = service.buildMessages(query, ctx, false);
    const last = msgs[msgs.length - 1];
    assert.equal(last.role, 'user');
    assert.equal(last.content, query);
  });

  test('with includeHistory=true history messages appear between system and user', () => {
    service.conversationHistory = [
      { role: 'user', content: 'Previous question' },
      { role: 'assistant', content: 'Previous answer' }
    ];
    const msgs = service.buildMessages(query, ctx, true);
    // Order: system, ...history, user
    assert.equal(msgs[0].role, 'system');
    assert.equal(msgs[1].role, 'user');
    assert.equal(msgs[1].content, 'Previous question');
    assert.equal(msgs[2].role, 'assistant');
    assert.equal(msgs[msgs.length - 1].role, 'user');
    assert.equal(msgs[msgs.length - 1].content, query);
    // Clean up
    service.conversationHistory = [];
  });

  test('with includeHistory=false only system + user messages present', () => {
    service.conversationHistory = [
      { role: 'user', content: 'Old question' },
      { role: 'assistant', content: 'Old answer' }
    ];
    const msgs = service.buildMessages(query, ctx, false);
    assert.equal(msgs.length, 2, 'should have exactly system + user (no history)');
    // Clean up
    service.conversationHistory = [];
  });
});

// ---------------------------------------------------------------------------
// chat (uses StubLLMProvider)
// ---------------------------------------------------------------------------
describe('ChatService.chat — StubLLMProvider', () => {
  let service;
  let stub;

  before(async () => {
    const doc = await loadDoc();
    stub = new StubLLMProvider('This is the stub answer.');
    service = new ChatService(stub);
    await service.initializeWithDocuments([doc]);
  });

  test('returns success:true with a response string', async () => {
    const result = await service.chat('What subjects are in the ISEB Common Pre-Test?');
    assert.equal(result.success, true);
    assert.equal(typeof result.response, 'string');
  });

  test('response matches what the stub returns', async () => {
    service.clearHistory();
    const result = await service.chat('What subjects are in the ISEB Common Pre-Test?', { includeHistory: false });
    assert.equal(result.response, 'This is the stub answer.');
  });

  test('sources array is populated with filename and chunkIndex', async () => {
    service.clearHistory();
    const result = await service.chat('What subjects are in the ISEB Common Pre-Test?', { includeHistory: false });
    assert.ok(Array.isArray(result.sources), 'sources should be an array');
    assert.ok(result.sources.length > 0, 'sources should not be empty');
    for (const src of result.sources) {
      assert.ok('filename' in src, 'source missing filename');
      assert.ok('chunkIndex' in src, 'source missing chunkIndex');
    }
  });

  test('conversationHistory grows by 2 after each chat call', async () => {
    service.clearHistory();
    const before = service.conversationHistory.length;
    await service.chat('Question A', { includeHistory: true });
    assert.equal(service.conversationHistory.length, before + 2);
  });

  test('calling chat twice adds 4 messages to history', async () => {
    service.clearHistory();
    await service.chat('Question A', { includeHistory: true });
    await service.chat('Question B', { includeHistory: true });
    assert.equal(service.conversationHistory.length, 4);
  });

  test('clearHistory resets conversationHistory to []', async () => {
    await service.chat('Question', { includeHistory: true });
    service.clearHistory();
    assert.equal(service.conversationHistory.length, 0);
  });

  test('fallback response returned when similarityThreshold is set to 0.99', async () => {
    service.clearHistory();
    const result = await service.chat('Any question', { similarityThreshold: 0.99, includeHistory: false });
    // At threshold 0.99 nothing should match; fallback has sources: []
    assert.equal(result.success, true);
    assert.equal(result.sources.length, 0);
    assert.ok(result.response.length > 0, 'fallback response should not be empty');
  });
});

// ---------------------------------------------------------------------------
// Full pipeline smoke test (real LLM — ClaudeLLMProvider, real embeddings)
// ---------------------------------------------------------------------------
describe('ChatService — full pipeline smoke test (real LLM)', () => {
  let service;

  before(async () => {
    const provider = new LocalEmbeddingProvider();
    const vectorStore = new VectorStore(provider);
    const llm = new ClaudeLLMProvider();

    service = new ChatService(llm);
    const doc = await DocumentLoader.loadSingleDocument(PDF_PATH);
    await service.initializeWithDocuments([doc]);
  });

  test('real LLM smoke test: success, non-empty response, correct source', async () => {
    const result = await service.chat(
      'What subjects are in the ISEB Common Pre-Test?',
      { includeHistory: false, maxTokens: 300 }
    );

    assert.equal(result.success, true, 'expected success:true');
    assert.equal(typeof result.response, 'string', 'response should be a string');
    assert.ok(result.response.length > 50, `response too short: "${result.response}"`);
    assert.ok(result.sources.length > 0, 'expected at least one source');
    assert.equal(result.sources[0].filename, '11-plus-syllabus.pdf');
  }, { timeout: 30000 });
});
