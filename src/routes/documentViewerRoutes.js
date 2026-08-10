const express = require('express');
const DocumentLoader = require('../core/DocumentLoader');
const VectorStore = require('../core/VectorStore');
const ChatService = require('../core/ChatService');
const { createChunkingStrategy } = require('../providers/ChunkingStrategy');
const path = require('path');

// Per-document chunking strategy config.
// ABRSM syllabus has repeating technical headers (PIANO, ARPEGGIOS) that don't
// map to meaningful sections — fixed-size chunking preserves grade content better.
// 11+ syllabus has clear subject headings (MATHEMATICS, ENGLISH) — section-aware
// keeps those isolated, improving retrieval quality.
const DOC_CHUNKING_STRATEGY = {
  'ABRSM_Piano_2025_2026_syllabus.pdf': 'grade',   // splits on GRADE N boundaries, prepends label to every chunk
  '11-plus-syllabus.pdf':               'section', // splits on subject headings (MATHEMATICS, ENGLISH etc.)
};

function getChunkingStrategyForDoc(filename) {
  const type = DOC_CHUNKING_STRATEGY[filename] || process.env.CHUNKING_STRATEGY || 'fixed';
  console.log(`📐 Chunking strategy for ${filename}: ${type}`);
  return createChunkingStrategy(type);
}

const router = express.Router();

// Cache for loaded documents to avoid reloading
let documentCache = null;
let cacheTimestamp = null;
const CACHE_DURATION = 5 * 60 * 1000; // 5 minutes

// Vector store cache for embeddings
let vectorStoreCache = null;
let vectorCacheTimestamp = null;
const VECTOR_CACHE_DURATION = 30 * 60 * 1000; // 30 minutes (embeddings are expensive)

// ChatService cache — keyed by filename so each doc gets its own initialized service
const chatServiceCache = {};   // { [filename]: { service, timestamp } }
const CHAT_CACHE_DURATION = 60 * 60 * 1000; // 1 hour (includes LLM setup)

// Retrieval defaults — single source of truth for topK across chat and vector-search routes
const DEFAULT_CHAT_TOP_K   = 6; // broader: covers pieces + scales + aural for "what to practice" queries
const DEFAULT_SEARCH_TOP_K = 3; // narrower: vector-search UI is for inspection, not full context assembly

/**
 * Get cached documents or reload if cache is stale
 */
async function getCachedDocuments() {
  const now = Date.now();

  if (!documentCache || !cacheTimestamp || (now - cacheTimestamp) > CACHE_DURATION) {
    console.log('🔄 Loading/refreshing document cache...');
    const documentsPath = path.join(__dirname, '../../public/pdf-sources');
    documentCache = await DocumentLoader.loadDocuments(documentsPath);
    cacheTimestamp = now;
    console.log(`✅ Cached ${documentCache.length} documents`);
  }

  return documentCache;
}

/**
 * GET /api/rag-docs - List all loaded documents
 */
router.get('/', async (req, res) => {
  try {
    const documents = await getCachedDocuments();
    const stats = DocumentLoader.getDocumentStats(documents);

    // Calculate total chunks across all documents
    let totalChunks = 0;
    for (const doc of documents) {
      const chunks = await DocumentLoader.chunkDocument(doc.content, 1000, 100);
      totalChunks += chunks.length;
    }
    stats.totalChunks = totalChunks;

    // Get embeddings info from cached ChatService if available
    let embeddingsInfo = null;
    if (chatServiceCache) {
      const vectorStats = chatServiceCache.vectorStore ? chatServiceCache.vectorStore.getStats() : null;
      if (vectorStats) {
        embeddingsInfo = {
          totalEmbeddings: vectorStats.totalEmbeddings,
          embeddingModel: vectorStats.model,
          maxChunksUsed: 10, // Based on ChatService config
          costOptimized: vectorStats.totalEmbeddings < totalChunks
        };
      }
    }

    // If no cached embeddings, show estimated info
    if (!embeddingsInfo) {
      embeddingsInfo = {
        totalEmbeddings: 0,
        embeddingModel: "text-embedding-3-small",
        maxChunksUsed: 10,
        costOptimized: true,
        estimated: true
      };
    }
    stats.embeddingsInfo = embeddingsInfo;

    const documentList = documents.map((doc, index) => ({
      id: index,
      filename: doc.metadata.filename,
      filesize: doc.metadata.filesize,
      pages: doc.metadata.pages,
      characters: doc.content.length,
      extractedAt: doc.metadata.extractedAt,
      preview: doc.content.substring(0, 200) + '...'
    }));

    res.json({
      success: true,
      stats,
      documents: documentList,
      cacheInfo: {
        cached: !!documentCache,
        cacheAge: cacheTimestamp ? Math.round((Date.now() - cacheTimestamp) / 1000) : 0
      }
    });
  } catch (error) {
    console.error('Error fetching documents:', error);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

/**
 * GET /api/rag-docs/test - Simple test endpoint
 */
router.get('/test', (req, res) => {
  res.json({
    success: true,
    message: 'RAG Documents API is working!',
    timestamp: new Date().toISOString(),
    server: 'Express'
  });
});

/**
 * GET /api/rag-docs/refresh - Force refresh document cache
 */
router.get('/refresh', async (req, res) => {
  try {
    // Clear cache to force reload
    documentCache = null;
    cacheTimestamp = null;

    const documents = await getCachedDocuments();
    res.json({
      success: true,
      message: `Refreshed ${documents.length} documents`,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

/**
 * GET /api/rag-docs/evaluate - Evaluate retrieval accuracy (HTML UI)
 */
router.get('/evaluate', async (req, res) => {
  try {
    const doc = req.query.doc;
    if (!doc) {
      return res.status(400).json({ success: false, error: 'doc query param required — e.g. /evaluate?doc=11-plus-syllabus.pdf' });
    }
    const chatService = await getCachedChatService(doc);

    console.log('📊 Running retrieval evaluation...');

    const evaluation = await chatService.evaluateRetrieval();

    // Return HTML UI if requested from browser, JSON otherwise
    const acceptsHTML = req.headers.accept && req.headers.accept.includes('text/html');

    if (acceptsHTML) {
      const html = `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG Retrieval Evaluation</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
            line-height: 1.6;
        }
        .container {
            background: white;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 30px;
            margin-bottom: 20px;
        }
        h1 {
            color: #2563eb;
            margin-top: 0;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }
        .metric-card {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
        }
        .metric-value {
            font-size: 2.5em;
            font-weight: bold;
            color: #1e40af;
            margin: 10px 0;
        }
        .metric-label {
            font-size: 0.9em;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .metric-desc {
            font-size: 0.8em;
            color: #475569;
            margin-top: 8px;
        }
        .difficulty-section {
            margin: 30px 0;
        }
        .difficulty-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
        }
        .difficulty-card {
            background: #fefefe;
            border: 1px solid #e5e7eb;
            border-radius: 6px;
            padding: 15px;
        }
        .difficulty-title {
            font-weight: 600;
            text-transform: capitalize;
            margin-bottom: 10px;
            color: #374151;
        }
        .difficulty-easy { border-left: 4px solid #10b981; }
        .difficulty-medium { border-left: 4px solid #f59e0b; }
        .difficulty-hard { border-left: 4px solid #ef4444; }

        .queries-section {
            margin-top: 40px;
        }
        .query-item {
            background: #f9fafb;
            border: 1px solid #e5e7eb;
            border-radius: 6px;
            padding: 15px;
            margin: 10px 0;
        }
        .query-question {
            font-weight: 600;
            color: #1f2937;
            margin-bottom: 8px;
        }
        .query-metrics {
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
            margin: 8px 0;
        }
        .query-metric {
            font-size: 0.9em;
            padding: 2px 8px;
            border-radius: 4px;
            background: #e5e7eb;
        }
        .good { background: #dcfce7; color: #166534; }
        .okay { background: #fef3c7; color: #92400e; }
        .poor { background: #fee2e2; color: #991b1b; }

        .nav {
            margin: 20px 0;
            text-align: center;
        }
        .nav a {
            color: #2563eb;
            text-decoration: none;
            margin: 0 15px;
            padding: 8px 16px;
            border: 1px solid #e2e8f0;
            border-radius: 6px;
            transition: all 0.2s;
        }
        .nav a:hover {
            background: #2563eb;
            color: white;
        }
        .timestamp {
            text-align: center;
            color: #6b7280;
            font-size: 0.9em;
            margin-top: 30px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 RAG Retrieval Evaluation</h1>

        <div class="nav">
            <a href="/api/rag-docs">← Back to Documents</a>
            <a href="/api/rag-docs/chat/test">Chat Test</a>
            <a href="/api/rag-docs/evaluate?format=json">View JSON</a>
        </div>

        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value">${evaluation.overall.avgPrecision}</div>
                <div class="metric-label">Precision</div>
                <div class="metric-desc">% of retrieved that was relevant</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">${evaluation.overall.avgRecall}</div>
                <div class="metric-label">Recall</div>
                <div class="metric-desc">% of relevant that was found</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">${evaluation.overall.avgF1}</div>
                <div class="metric-label">F1-Score</div>
                <div class="metric-desc">Combined precision & recall</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">${evaluation.overall.avgMRR}</div>
                <div class="metric-label">MRR</div>
                <div class="metric-desc">First result relevance</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">${evaluation.overall.avgNDCG || 'N/A'}</div>
                <div class="metric-label">NDCG</div>
                <div class="metric-desc">Ranking quality</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">${evaluation.overall.avgRetrievalTime || 'N/A'}ms</div>
                <div class="metric-label">Speed</div>
                <div class="metric-desc">Average retrieval time</div>
            </div>
        </div>

        <div class="difficulty-section">
            <h2>Performance by Difficulty</h2>
            <div class="difficulty-grid">
                ${Object.entries(evaluation.byDifficulty).map(([difficulty, metrics]) => `
                    <div class="difficulty-card difficulty-${difficulty}">
                        <div class="difficulty-title">${difficulty} Questions</div>
                        <div><strong>F1:</strong> ${metrics.avgF1} <strong>MRR:</strong> ${metrics.avgMRR}</div>
                        <div><strong>Count:</strong> ${metrics.count}</div>
                    </div>
                `).join('')}
            </div>
        </div>

        <div class="queries-section">
            <h2>Individual Query Results</h2>
            ${evaluation.detailed.map(query => `
                <div class="query-item">
                    <div class="query-question">${query.question}</div>
                    <div class="query-metrics">
                        <span class="query-metric ${query.precision > 0.7 ? 'good' : query.precision > 0.4 ? 'okay' : 'poor'}">
                            Precision: ${query.precision}
                        </span>
                        <span class="query-metric ${query.recall > 0.7 ? 'good' : query.recall > 0.4 ? 'okay' : 'poor'}">
                            Recall: ${query.recall}
                        </span>
                        <span class="query-metric ${query.f1Score > 0.7 ? 'good' : query.f1Score > 0.4 ? 'okay' : 'poor'}">
                            F1: ${query.f1Score}
                        </span>
                        <span class="query-metric">
                            Difficulty: ${query.difficulty}
                        </span>
                        <span class="query-metric">
                            Retrieved: ${query.retrievedCount}
                        </span>
                    </div>
                    ${query.topResult ? `
                        <div style="margin-top: 10px; padding: 10px; background: #f3f4f6; border-radius: 4px; font-size: 0.9em;">
                            <strong>Top Result:</strong> ${query.topResult.metadata.filename}
                            (similarity: ${(query.topResult.similarity * 100).toFixed(1)}%)
                            <div style="margin-top: 8px; padding: 8px; background: #ffffff; border-radius: 4px; font-size: 0.85em; max-height: 150px; overflow-y: auto; border-left: 3px solid #3b82f6;">
                                <strong>Content Preview:</strong><br>
                                ${query.topResult.content.substring(0, 300)}${query.topResult.content.length > 300 ? '...' : ''}
                            </div>
                        </div>
                    ` : ''}
                    ${query.retrievedSources && query.retrievedSources.length > 1 ? `
                        <details style="margin-top: 10px;">
                            <summary style="cursor: pointer; font-weight: 600; color: #374151;">
                                View All ${query.retrievedSources.length} Retrieved Results
                            </summary>
                            <div style="margin-top: 10px; max-height: 300px; overflow-y: auto;">
                                ${query.retrievedSources.map((source, idx) => `
                                    <div style="margin: 8px 0; padding: 8px; background: #f9fafb; border-radius: 4px; border-left: 3px solid ${idx === 0 ? '#10b981' : '#6b7280'};">
                                        <div style="font-weight: 600; font-size: 0.9em;">
                                            #${idx + 1} - ${source.filename} (chunk ${source.chunkIndex})
                                            <span style="color: #6b7280; font-weight: normal;">
                                                - ${(source.similarity * 100).toFixed(1)}% match
                                            </span>
                                        </div>
                                        ${source.content ? `
                                            <div style="margin-top: 4px; font-size: 0.8em; color: #4b5563; max-height: 80px; overflow: hidden;">
                                                ${source.content.substring(0, 200)}${source.content.length > 200 ? '...' : ''}
                                            </div>
                                        ` : ''}
                                    </div>
                                `).join('')}
                            </div>
                        </details>
                    ` : ''}
                </div>
            `).join('')}
        </div>

        <div class="timestamp">
            Evaluation completed at: ${new Date().toLocaleString()}
        </div>
    </div>
</body>
</html>`;

      res.set('Content-Type', 'text/html');
      res.send(html);
    } else {
      // Return JSON for API calls
      res.json({
        success: true,
        evaluation: evaluation,
        timestamp: new Date().toISOString()
      });
    }

  } catch (error) {
    console.error('❌ Evaluation error:', error.message);
    res.status(500).json({
      success: false,
      error: error.message,
      helpText: 'Make sure ChatService is properly initialized'
    });
  }
});

/**
 * GET /api/rag-docs/:id - View specific document content
 */
router.get('/:id', async (req, res) => {
  try {
    const documents = await getCachedDocuments();
    const docId = parseInt(req.params.id);

    if (isNaN(docId) || docId < 0 || docId >= documents.length) {
      return res.status(404).json({
        success: false,
        error: 'Document not found'
      });
    }

    const document = documents[docId];

    res.json({
      success: true,
      document: {
        id: docId,
        metadata: document.metadata,
        contentLength: document.content.length,
        content: document.content
      }
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

/**
 * GET /api/rag-docs/:id/chunks - View document chunks
 */
router.get('/:id/chunks', async (req, res) => {
  try {
    const documents = await getCachedDocuments();
    const docId = parseInt(req.params.id);

    if (isNaN(docId) || docId < 0 || docId >= documents.length) {
      return res.status(404).json({
        success: false,
        error: 'Document not found'
      });
    }

    const document = documents[docId];

    // Get chunk parameters from query string
    const chunkSize = parseInt(req.query.size) || 1000;
    const overlap = parseInt(req.query.overlap) || 100;

    console.log(`📝 Chunking document ${docId} with size=${chunkSize}, overlap=${overlap}`);
    const chunks = await DocumentLoader.chunkDocument(document.content, chunkSize, overlap);

    res.json({
      success: true,
      document: {
        id: docId,
        filename: document.metadata.filename,
        contentLength: document.content.length
      },
      chunkConfig: {
        size: chunkSize,
        overlap: overlap
      },
      chunksCount: chunks.length,
      chunks: chunks.map((chunk, index) => ({
        id: index,
        content: chunk.content,
        length: chunk.length,
        startChar: chunk.startChar,
        endChar: chunk.endChar,
        preview: chunk.content.substring(0, 100) + (chunk.content.length > 100 ? '...' : '')
      }))
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

/**
 * GET /api/rag-docs/:id/view - HTML view of document content
 */
router.get('/:id/view', async (req, res) => {
  try {
    const documents = await getCachedDocuments();
    const docId = parseInt(req.params.id);

    if (isNaN(docId) || docId < 0 || docId >= documents.length) {
      return res.status(404).send('<h1>Document not found</h1>');
    }

    const document = documents[docId];

    // Create HTML view
    const html = `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>${document.metadata.filename} - Document Viewer</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
            background: #f5f5f5;
        }
        .header {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .metadata {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 15px 0;
        }
        .metadata-item {
            background: #f8f9fa;
            padding: 10px;
            border-radius: 4px;
        }
        .metadata-label {
            font-weight: bold;
            color: #666;
            font-size: 0.9em;
        }
        .content {
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            white-space: pre-wrap;
            font-family: 'Georgia', serif;
            font-size: 16px;
            line-height: 1.8;
        }
        .nav {
            margin: 20px 0;
            text-align: center;
        }
        .nav a {
            display: inline-block;
            padding: 10px 20px;
            background: #007bff;
            color: white;
            text-decoration: none;
            border-radius: 5px;
            margin: 0 10px;
        }
        .nav a:hover {
            background: #0056b3;
        }
        .search-box {
            margin: 15px 0;
        }
        .search-box input {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 16px;
        }
        .highlight {
            background-color: yellow;
            padding: 2px 4px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>📄 ${document.metadata.filename}</h1>

        <div class="metadata">
            <div class="metadata-item">
                <div class="metadata-label">File Size</div>
                <div>${Math.round(document.metadata.filesize / 1024)} KB</div>
            </div>
            <div class="metadata-item">
                <div class="metadata-label">Pages</div>
                <div>${document.metadata.pages}</div>
            </div>
            <div class="metadata-item">
                <div class="metadata-label">Characters</div>
                <div>${document.content.length.toLocaleString()}</div>
            </div>
            <div class="metadata-item">
                <div class="metadata-label">Extracted</div>
                <div>${new Date(document.metadata.extractedAt).toLocaleString()}</div>
            </div>
        </div>

        <div class="search-box">
            <input type="text" id="searchInput" placeholder="Search in document..." onkeyup="searchText()">
        </div>

        <div class="nav">
            <a href="/api/rag-docs">← Back to Documents</a>
            <a href="/api/rag-docs/${docId}/chunks">View Chunks</a>
            <a href="/api/rag-docs/${docId}">JSON Data</a>
        </div>
    </div>

    <div class="content" id="documentContent">${document.content}</div>

    <script>
        function searchText() {
            const searchTerm = document.getElementById('searchInput').value;
            const content = document.getElementById('documentContent');
            const originalText = \`${document.content.replace(/`/g, '\\`')}\`;

            if (searchTerm.length < 3) {
                content.innerHTML = originalText;
                return;
            }

            const regex = new RegExp(searchTerm, 'gi');
            const highlightedText = originalText.replace(regex, '<span class="highlight">$&</span>');
            content.innerHTML = highlightedText;

            // Scroll to first match
            const firstMatch = content.querySelector('.highlight');
            if (firstMatch) {
                firstMatch.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
        }
    </script>
</body>
</html>`;

    res.set('Content-Type', 'text/html');
    res.send(html);

  } catch (error) {
    res.status(500).send(`<h1>Error</h1><p>${error.message}</p>`);
  }
});

/**
 * GET /api/rag-docs/search - Search across all documents
 */
router.get('/search/:query', async (req, res) => {
  try {
    const documents = await getCachedDocuments();
    const query = req.params.query.toLowerCase();
    const results = [];

    documents.forEach((doc, docIndex) => {
      const content = doc.content.toLowerCase();
      let index = content.indexOf(query);

      while (index !== -1) {
        const start = Math.max(0, index - 100);
        const end = Math.min(content.length, index + query.length + 100);
        const context = doc.content.substring(start, end);

        results.push({
          documentId: docIndex,
          filename: doc.metadata.filename,
          position: index,
          context: context,
          preview: context.substring(0, 200) + '...'
        });

        index = content.indexOf(query, index + 1);
      }
    });

    res.json({
      success: true,
      query: req.params.query,
      resultsCount: results.length,
      results: results.slice(0, 50) // Limit to 50 results
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

/**
 * Get cached vector store or create/refresh if needed
 */
async function getCachedVectorStore() {
  const now = Date.now();

  if (!vectorStoreCache || !vectorCacheTimestamp || (now - vectorCacheTimestamp) > VECTOR_CACHE_DURATION) {
    console.log('🔮 Creating/refreshing vector store cache...');

    // Get documents first
    const documents = await getCachedDocuments();

    if (documents.length === 0) {
      throw new Error('No documents available for vector store');
    }

    // Initialize vector store
    vectorStoreCache = new VectorStore();

    for (const document of documents) {
      console.log(`🔮 Processing embeddings for: ${document.metadata.filename}`);

      const chunks = await DocumentLoader.chunkDocument(document.content, 1000, 100);
      const limitedChunks = chunks;

      const embeddings = await vectorStoreCache.generateEmbeddings(limitedChunks, document.metadata);
      vectorStoreCache.addEmbeddings(embeddings);
    }

    vectorCacheTimestamp = now;
    console.log(`✅ Vector store cached with ${vectorStoreCache.getStats().totalEmbeddings} embeddings`);
  }

  return vectorStoreCache;
}

/**
 * POST /api/rag-docs/vector-search - Perform vector similarity search
 */
router.post('/vector-search', async (req, res) => {
  try {
    const { query, doc, topK = DEFAULT_SEARCH_TOP_K, threshold = 0.1 } = req.body;

    if (!query || typeof query !== 'string' || query.trim().length === 0) {
      return res.status(400).json({
        success: false,
        error: 'Query is required and must be a non-empty string'
      });
    }

    if (!doc) {
      return res.status(400).json({
        success: false,
        error: 'doc (filename) is required — select a document from the dashboard first'
      });
    }

    console.log(`🔍 Vector search [${doc}]: "${query}"`);

    // Reuse the per-doc ChatService cache so we never embed the wrong docs
    const chatService = await getCachedChatService(doc);
    const vectorStore = chatService.vectorStore;

    // Perform search
    const results = await vectorStore.similaritySearch(query.trim(), topK, threshold);

    // Get vector store stats
    const stats = vectorStore.getStats();

    res.json({
      success: true,
      query: query.trim(),
      searchParams: {
        topK,
        threshold
      },
      resultsCount: results.length,
      results: results.map(result => ({
        id: result.id,
        content: result.content,
        similarity: result.similarity,
        preview: result.content.substring(0, 200) + (result.content.length > 200 ? '...' : ''),
        metadata: {
          filename: result.metadata.filename,
          chunkIndex: result.metadata.chunkIndex,
          startChar: result.metadata.startChar,
          endChar: result.metadata.endChar
        }
      })),
      vectorStoreStats: stats,
      processingTime: Date.now() - Date.now() // Will be calculated properly in real implementation
    });

  } catch (error) {
    console.error('❌ Vector search error:', error.message);
    res.status(500).json({
      success: false,
      error: error.message,
      helpText: 'Make sure OPENAI_API_KEY is set and you have API credits'
    });
  }
});

/**
 * GET /api/rag-docs/vector-search/test - Get test interface
 */
router.get('/vector-search/test', (req, res) => {
  const doc = req.query.doc || '';

  const exampleQueries = doc.includes('11-plus') ? [
    { label: 'ISEB Pre-Test subjects',  query: 'What subjects are included in the ISEB Common Pre-Test?' },
    { label: 'Maths paper duration',    query: 'How long is the mathematics paper and is a calculator allowed?' },
    { label: 'Maths paper sections',    query: 'What are the three sections of the maths paper?' },
    { label: 'Topics not tested',       query: 'What topics are not tested in the maths exam?' },
    { label: 'Composition forms',       query: 'What writing forms are accepted in the composition paper?' },
  ] : doc.includes('ABRSM') ? [
    { label: 'Piano scales',            query: 'What are piano scales?' },
    { label: 'How many grades',         query: 'How many grades are there?' },
    { label: 'ABRSM requirements',      query: 'ABRSM requirements' },
    { label: 'Practice techniques',     query: 'Practice techniques' },
    { label: 'Musical theory',          query: 'Musical theory' },
  ] : [
    { label: 'ISEB Pre-Test subjects',  query: 'What subjects are included in the ISEB Common Pre-Test?' },
    { label: 'Maths paper sections',    query: 'What are the three sections of the maths paper?' },
    { label: 'Topics not tested',       query: 'What topics are not tested in the maths exam?' },
  ];

  const exampleChipsHtml = exampleQueries.map(e =>
    `<span class="example-query" onclick="setQuery('${e.query}')">${e.label}</span>`
  ).join('\n            ');

  const html = `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Vector Search Test</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .search-box {
            margin-bottom: 20px;
        }
        .search-box input[type="text"] {
            width: 70%;
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: 4px;
            font-size: 16px;
        }
        .search-box button {
            width: 25%;
            padding: 12px;
            background: #007bff;
            color: white;
            border: none;
            border-radius: 4px;
            font-size: 16px;
            cursor: pointer;
            margin-left: 10px;
        }
        .search-box button:hover {
            background: #0056b3;
        }
        .search-box button:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        .controls {
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
        }
        .control-group {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .control-group label {
            font-weight: bold;
        }
        .control-group input {
            padding: 5px;
            border: 1px solid #ddd;
            border-radius: 3px;
            width: 80px;
        }
        .results {
            margin-top: 20px;
        }
        .result-item {
            background: #f8f9fa;
            padding: 15px;
            margin-bottom: 15px;
            border-radius: 6px;
            border-left: 4px solid #007bff;
        }
        .result-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        .similarity-score {
            background: #007bff;
            color: white;
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: bold;
        }
        .result-content {
            line-height: 1.6;
            margin-bottom: 10px;
        }
        .result-metadata {
            font-size: 12px;
            color: #666;
            display: flex;
            gap: 15px;
        }
        .loading {
            text-align: center;
            padding: 20px;
            color: #666;
        }
        .error {
            background: #f8d7da;
            color: #721c24;
            padding: 15px;
            border-radius: 4px;
            margin: 20px 0;
        }
        .stats {
            background: #e7f3ff;
            padding: 15px;
            border-radius: 4px;
            margin-bottom: 20px;
            font-size: 14px;
        }
        .example-queries {
            margin: 20px 0;
        }
        .example-query {
            display: inline-block;
            background: #e9ecef;
            padding: 5px 10px;
            margin: 5px;
            border-radius: 15px;
            cursor: pointer;
            font-size: 14px;
        }
        .example-query:hover {
            background: #dee2e6;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 Vector Search${doc ? ' — ' + doc : ''}</h1>
        <p>Test the RAG vector search functionality with real OpenAI embeddings</p>

        <div class="search-box">
            <input type="text" id="searchQuery" placeholder="Enter your search query..." value="${exampleQueries[0].query}">
            <button onclick="performSearch()" id="searchBtn">Search</button>
        </div>

        <div class="controls">
            <div class="control-group">
                <label>Top K:</label>
                <input type="number" id="topK" value="${DEFAULT_SEARCH_TOP_K}" min="1" max="10">
            </div>
            <div class="control-group">
                <label>Threshold:</label>
                <input type="number" id="threshold" value="0.1" min="0" max="1" step="0.1">
            </div>
        </div>

        <div class="example-queries">
            <strong>Example queries:</strong><br>
            ${exampleChipsHtml}
        </div>

        <div id="results"></div>
    </div>

    <script>
        async function performSearch() {
            const query = document.getElementById('searchQuery').value.trim();
            const topK = parseInt(document.getElementById('topK').value);
            const threshold = parseFloat(document.getElementById('threshold').value);
            const searchBtn = document.getElementById('searchBtn');
            const resultsDiv = document.getElementById('results');

            if (!query) {
                alert('Please enter a search query');
                return;
            }

            // Show loading state
            searchBtn.disabled = true;
            searchBtn.textContent = 'Searching...';
            resultsDiv.innerHTML = '<div class="loading">🔮 Generating embeddings and searching...</div>';

            try {
                const response = await fetch('/api/rag-docs/vector-search', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ query, topK, threshold, doc: new URLSearchParams(window.location.search).get('doc') })
                });

                const data = await response.json();

                if (data.success) {
                    displayResults(data);
                } else {
                    displayError(data.error, data.helpText);
                }

            } catch (error) {
                displayError('Network error: ' + error.message);
            } finally {
                searchBtn.disabled = false;
                searchBtn.textContent = 'Search';
            }
        }

        function displayResults(data) {
            const resultsDiv = document.getElementById('results');

            let html = \`
                <div class="stats">
                    📊 <strong>Search Results:</strong> \${data.resultsCount} matches for "\${data.query}" |
                    Vector Store: \${data.vectorStoreStats.totalEmbeddings} embeddings from \${data.vectorStoreStats.uniqueDocuments} documents
                </div>
            \`;

            if (data.results.length === 0) {
                html += '<p>No results found above the similarity threshold. Try lowering the threshold or using different keywords.</p>';
            } else {
                data.results.forEach((result, index) => {
                    html += \`
                        <div class="result-item">
                            <div class="result-header">
                                <strong>Result \${index + 1}</strong>
                                <span class="similarity-score">\${(result.similarity * 100).toFixed(1)}% match</span>
                            </div>
                            <div class="result-content">\${result.preview}</div>
                            <div class="result-metadata">
                                <span>📄 \${result.metadata.filename}</span>
                                <span>📝 Chunk \${result.metadata.chunkIndex}</span>
                                <span>📍 Chars \${result.metadata.startChar}-\${result.metadata.endChar}</span>
                            </div>
                        </div>
                    \`;
                });
            }

            resultsDiv.innerHTML = html;
        }

        function displayError(error, helpText = '') {
            const resultsDiv = document.getElementById('results');
            resultsDiv.innerHTML = \`
                <div class="error">
                    <strong>Error:</strong> \${error}
                    \${helpText ? \`<br><br><strong>Help:</strong> \${helpText}\` : ''}
                </div>
            \`;
        }

        function setQuery(query) {
            document.getElementById('searchQuery').value = query;
        }

        // Allow Enter key to trigger search
        document.getElementById('searchQuery').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                performSearch();
            }
        });
    </script>
</body>
</html>`;

  res.set('Content-Type', 'text/html');
  res.send(html);
});

/**
 * Get cached ChatService for a specific document, or create one if stale/missing.
 * @param {string} filename - e.g. '11-plus-syllabus.pdf'
 */
async function getCachedChatService(filename) {
  const now = Date.now();
  const cached = chatServiceCache[filename];

  if (cached && (now - cached.timestamp) < CHAT_CACHE_DURATION) {
    return cached.service;
  }

  console.log(`🤖 Initializing ChatService for: ${filename}`);

  const allDocuments = await getCachedDocuments();
  const doc = allDocuments.find(d => d.metadata.filename === filename);
  if (!doc) throw new Error(`Document not found: ${filename}`);

  const service = new ChatService();
  await service.initializeWithDocuments([doc], {
    chunkSize: 1000,
    chunkOverlap: 100,
    chunkingStrategy: getChunkingStrategyForDoc(filename)
  });

  chatServiceCache[filename] = { service, timestamp: now };
  const stats = service.getConversationStats();
  console.log(`✅ ChatService ready for ${filename} — ${stats.vectorStoreStats.totalEmbeddings} embeddings`);

  return service;
}

/**
 * POST /api/rag-docs/chat - Chat with RAG pipeline
 */
router.post('/chat', async (req, res) => {
  try {
    const {
      message,
      doc,
      includeHistory = true,
      temperature = 0.7,
      topK = DEFAULT_CHAT_TOP_K,
      clearHistory = false
    } = req.body;

    if (!message || typeof message !== 'string' || message.trim().length === 0) {
      return res.status(400).json({
        success: false,
        error: 'Message is required and must be a non-empty string'
      });
    }

    if (!doc) {
      return res.status(400).json({
        success: false,
        error: 'doc (filename) is required — select a document from the dashboard first'
      });
    }

    console.log(`💬 Chat request for [${doc}]: "${message}"`);

    // Get chat service scoped to this document
    const chatService = await getCachedChatService(doc);

    // Clear history if requested
    if (clearHistory) {
      chatService.clearHistory();
    }

    // Generate response
    const startTime = Date.now();
    const response = await chatService.chat(message.trim(), {
      includeHistory,
      temperature,
      topK,
      similarityThreshold: 0.1,
      maxTokens: 1000
    });

    const processingTime = Date.now() - startTime;

    res.json({
      success: true,
      message: response.response,
      sources: response.sources,
      metadata: {
        ...response.metadata,
        processingTime,
        conversationLength: response.conversationLength
      },
      conversationStats: chatService.getConversationStats()
    });

  } catch (error) {
    console.error('❌ Chat error:', error.message);
    res.status(500).json({
      success: false,
      error: error.message,
      helpText: 'Make sure OPENAI_API_KEY is set and you have API credits'
    });
  }
});

// renderMarkdown is defined as a plain string so it can be injected into the HTML
// template literal without backslash-escaping issues. Regex literals and \n inside
// the function are literal source code characters — not interpreted by Node.js.
const renderMarkdownFn = `
function renderMarkdown(text) {
  if (!text) return '';
  var escHtml = function(s) {
    return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  };
  var lines = text.split('\\n');
  var out = [];
  var i = 0;
  while (i < lines.length) {
    var line = lines[i];
    if (/^### /.test(line)) { out.push('<h3>' + escHtml(line.slice(4)) + '</h3>'); i++; continue; }
    if (/^## /.test(line))  { out.push('<h2>' + escHtml(line.slice(3)) + '</h2>'); i++; continue; }
    if (/^# /.test(line))   { out.push('<h1>' + escHtml(line.slice(2)) + '</h1>'); i++; continue; }
    if (/^[-*] /.test(line)) {
      var items = [];
      while (i < lines.length && /^[-*] /.test(lines[i])) {
        items.push('<li>' + escHtml(lines[i].slice(2)) + '</li>');
        i++;
      }
      out.push('<ul>' + items.join('') + '</ul>');
      continue;
    }
    if (/^\\d+\\. /.test(line)) {
      var items = [];
      while (i < lines.length && /^\\d+\\. /.test(lines[i])) {
        items.push('<li>' + escHtml(lines[i].replace(/^\\d+\\. /, '')) + '</li>');
        i++;
      }
      out.push('<ol>' + items.join('') + '</ol>');
      continue;
    }
    if (line.trim() === '') { out.push('<br>'); i++; continue; }
    out.push(escHtml(line));
    i++;
  }
  var html = out.join('\\n');
  html = html.replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>');
  html = html.replace(/\\*([^*\\n]+?)\\*/g, '<em>$1</em>');
  html = html.replace(/([^>])\\n([^<])/g, '$1<br>$2');
  return '<div class="md">' + html + '</div>';
}
`;

/**
 * GET /api/rag-docs/chat/test - Get chat test interface
 */
router.get('/chat/test', (req, res) => {
  const doc = req.query.doc || '';

  const exampleMessages = doc.includes('11-plus') ? [
    { label: 'ISEB Pre-Test subjects',   msg: 'What subjects are included in the ISEB Common Pre-Test?' },
    { label: 'Maths paper format',       msg: 'How long is the mathematics paper and is a calculator allowed?' },
    { label: 'Maths sections',           msg: 'What are the three sections of the maths paper?' },
    { label: 'Topics not tested',        msg: 'What topics are not tested in the maths exam?' },
    { label: 'Composition forms',        msg: 'What writing forms are accepted in the composition paper?' },
  ] : doc.includes('ABRSM') ? [
    { label: 'Piano scales',             msg: 'What are piano scales?' },
    { label: 'How many grades',          msg: 'How many grades are there?' },
    { label: 'Grade 1 practice',         msg: 'What should I practice for Grade 1?' },
    { label: 'ABRSM requirements',       msg: 'Tell me about ABRSM requirements' },
  ] : [
    { label: 'ISEB Pre-Test subjects',   msg: 'What subjects are included in the ISEB Common Pre-Test?' },
    { label: 'Maths sections',           msg: 'What are the three sections of the maths paper?' },
    { label: 'Topics not tested',        msg: 'What topics are not tested in the maths exam?' },
  ];

  const exampleBtnsHtml = exampleMessages.map(e =>
    `<button class="example-btn" onclick="setMessage('${e.msg}')">${e.label}</button>`
  ).join('\n                ');

  const welcomeHint = doc.includes('11-plus')
    ? '💡 Try asking about the 11+ exam subjects, maths sections, or English paper format'
    : doc.includes('ABRSM')
    ? '💡 Try asking about piano scales, ABRSM requirements, or practice techniques'
    : '💡 Ask a question about your selected document';

  const html = `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG Chat Test</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .container {
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            overflow: hidden;
            height: 80vh;
            display: flex;
            flex-direction: column;
        }
        .chat-header {
            background: #007bff;
            color: white;
            padding: 20px;
            text-align: center;
        }
        .chat-messages {
            flex: 1;
            padding: 20px;
            overflow-y: auto;
            background: #f8f9fa;
        }
        .message {
            margin-bottom: 20px;
            padding: 15px;
            border-radius: 8px;
            max-width: 80%;
        }
        .message.user {
            background: #007bff;
            color: white;
            margin-left: auto;
            text-align: right;
        }
        .message.assistant {
            background: white;
            border: 1px solid #dee2e6;
            margin-right: auto;
        }
        .message-content {
            line-height: 1.6;
        }
        .message-meta {
            font-size: 12px;
            opacity: 0.7;
            margin-top: 8px;
        }
        .sources {
            margin-top: 10px;
            padding-top: 10px;
            border-top: 1px solid #e9ecef;
        }
        .source-item {
            background: #e7f3ff;
            padding: 8px;
            margin: 5px 0;
            border-radius: 4px;
            font-size: 12px;
        }
        .chat-input {
            padding: 20px;
            background: white;
            border-top: 1px solid #dee2e6;
        }
        .input-group {
            display: flex;
            gap: 10px;
            margin-bottom: 10px;
        }
        .input-group input {
            flex: 1;
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: 4px;
            font-size: 16px;
        }
        .input-group button {
            padding: 12px 24px;
            background: #007bff;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
        }
        .input-group button:hover {
            background: #0056b3;
        }
        .input-group button:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        .controls {
            display: flex;
            gap: 15px;
            align-items: center;
            font-size: 14px;
        }
        .controls label {
            display: flex;
            align-items: center;
            gap: 5px;
        }
        .controls input[type="checkbox"] {
            margin: 0;
        }
        .controls input[type="range"] {
            width: 80px;
        }
        .loading {
            text-align: center;
            padding: 20px;
            color: #666;
        }
        .error {
            background: #f8d7da;
            color: #721c24;
            padding: 15px;
            border-radius: 4px;
            margin: 10px 0;
        }
        .example-questions {
            margin: 10px 0;
        }
        .example-btn {
            background: #e9ecef;
            border: none;
            padding: 5px 10px;
            margin: 2px;
            border-radius: 15px;
            cursor: pointer;
            font-size: 12px;
        }
        .example-btn:hover {
            background: #dee2e6;
        }
        .message-content h1, .message-content h2, .message-content h3 {
            margin: 10px 0 4px;
            font-weight: 600;
            line-height: 1.3;
        }
        .message-content h1 { font-size: 1.1em; }
        .message-content h2 { font-size: 1.05em; }
        .message-content h3 { font-size: 1em; color: #374151; }
        .message-content ul, .message-content ol {
            margin: 4px 0 4px 18px;
            padding: 0;
        }
        .message-content li { margin: 2px 0; }
        .message-content p { margin: 4px 0; }
        .message-content strong { font-weight: 600; }
    </style>
</head>
<body>
    <div class="container">
        <div class="chat-header">
            <h1>🤖 RAG Chat${doc ? ' — ' + doc : ''}</h1>
            <p>Ask questions about your documents using AI</p>
        </div>

        <div class="chat-messages" id="chatMessages">
            <div class="message assistant">
                <div class="message-content">
                    Hello! I'm ready to answer questions about your documents. What would you like to know?
                </div>
                <div class="message-meta">
                    ${welcomeHint}
                </div>
            </div>
        </div>

        <div class="chat-input">
            <div class="example-questions">
                <strong>Quick examples:</strong><br>
                ${exampleBtnsHtml}
            </div>

            <div class="input-group">
                <input type="text" id="messageInput" placeholder="Ask a question about your documents..." onkeypress="handleKeyPress(event)">
                <button onclick="sendMessage()" id="sendBtn">Send</button>
            </div>

            <div class="controls">
                <label>
                    <input type="checkbox" id="includeHistory" checked> Include conversation history
                </label>
                <label>
                    Temperature: <input type="range" id="temperature" min="0" max="1" step="0.1" value="0.7">
                    <span id="tempValue">0.7</span>
                </label>
                <button onclick="clearChat()" style="padding: 5px 10px; font-size: 12px; background: #6c757d;">Clear Chat</button>
            </div>
        </div>
    </div>

    <script>
        let isLoading = false;

        // Update temperature display
        document.getElementById('temperature').addEventListener('input', function(e) {
            document.getElementById('tempValue').textContent = e.target.value;
        });

        async function sendMessage() {
            const messageInput = document.getElementById('messageInput');
            const message = messageInput.value.trim();

            if (!message || isLoading) return;

            // Add user message to chat
            addMessage('user', message);
            messageInput.value = '';

            // Show loading
            const loadingId = addMessage('assistant', '', true);
            isLoading = true;
            updateSendButton(false);

            try {
                const response = await fetch('/api/rag-docs/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        message: message,
                        doc: new URLSearchParams(window.location.search).get('doc'),
                        includeHistory: document.getElementById('includeHistory').checked,
                        temperature: parseFloat(document.getElementById('temperature').value),
                        topK: ${DEFAULT_CHAT_TOP_K}
                    })
                });

                const data = await response.json();

                // Remove loading message
                document.getElementById(loadingId).remove();

                if (data.success) {
                    addMessage('assistant', data.message, false, {
                        sources: data.sources,
                        metadata: data.metadata
                    });
                } else {
                    addMessage('assistant', \`Error: \${data.error}\`, false, { error: true });
                }

            } catch (error) {
                document.getElementById(loadingId).remove();
                addMessage('assistant', \`Network error: \${error.message}\`, false, { error: true });
            } finally {
                isLoading = false;
                updateSendButton(true);
            }
        }

        function scoreColor(score) {
            if (score >= 0.75) return '#10b981'; // green
            if (score >= 0.5)  return '#f59e0b'; // amber
            return '#ef4444';                     // red
        }

        function scoreBar(score) {
            const pct = Math.round(score * 100);
            const color = scoreColor(score);
            return \`<div style="display:flex;align-items:center;gap:6px;">
                <div style="flex:1;background:#e5e7eb;border-radius:4px;height:6px;">
                    <div style="width:\${pct}%;background:\${color};height:6px;border-radius:4px;"></div>
                </div>
                <span style="font-size:0.8em;color:\${color};font-weight:600;min-width:32px;">\${pct}%</span>
            </div>\`;
        }

        ${renderMarkdownFn}

        function addMessage(role, content, isLoading = false, metadata = {}) {
            const messagesDiv = document.getElementById('chatMessages');
            const messageDiv = document.createElement('div');
            const messageId = 'msg_' + Date.now() + Math.random().toString(36).substr(2, 9);
            messageDiv.id = messageId;
            messageDiv.className = \`message \${role}\`;

            let messageHTML = \`<div class="message-content">\`;

            if (isLoading) {
                messageHTML += \`<div class="loading">🔮 Thinking...</div>\`;
            } else {
                messageHTML += renderMarkdown(content);
            }

            messageHTML += \`</div>\`;

            if (role === 'assistant' && !isLoading && metadata.sources) {
                const sources   = metadata.sources || [];
                const meta      = metadata.metadata || {};
                const procTime  = meta.processingTime || 0;
                const chunks    = sources.length;
                const topScore  = chunks > 0 ? sources[0].similarity : 0;
                const avgScore  = chunks > 0
                    ? sources.reduce((s, r) => s + r.similarity, 0) / chunks
                    : 0;

                // ── Retrieval metrics computed from similarity scores ────────
                // Threshold for "relevant": similarity >= 0.5
                const RELEVANT_THRESHOLD = 0.5;
                const relevantChunks = sources.filter(s => s.similarity >= RELEVANT_THRESHOLD);

                // Precision@K: fraction of retrieved chunks above threshold
                const precisionAtK = chunks > 0 ? relevantChunks.length / chunks : 0;

                // MRR: 1 / rank of first chunk above threshold (rank is 1-based)
                let mrr = 0;
                for (let i = 0; i < sources.length; i++) {
                    if (sources[i].similarity >= RELEVANT_THRESHOLD) { mrr = 1 / (i + 1); break; }
                }

                // NDCG@K: normalised discounted cumulative gain using similarity as relevance
                // relevance score = 1 if >= threshold, else 0
                let dcg = 0;
                for (let i = 0; i < sources.length; i++) {
                    const rel = sources[i].similarity >= RELEVANT_THRESHOLD ? 1 : 0;
                    dcg += rel / Math.log2(i + 2); // log2(rank+1), rank is 1-based so i+2
                }
                // Ideal DCG: all relevant chunks ranked first
                let idcg = 0;
                for (let i = 0; i < relevantChunks.length; i++) {
                    idcg += 1 / Math.log2(i + 2);
                }
                const ndcg = idcg > 0 ? dcg / idcg : (chunks > 0 && relevantChunks.length === 0 ? 0 : 1);

                function metricBadge(label, value) {
                    const pct = Math.round(value * 100);
                    const color = scoreColor(value);
                    return \`<div style="background:#f8fafc;border:1px solid #e5e7eb;border-radius:6px;padding:6px 10px;text-align:center;">
                        <div style="font-size:1.1em;font-weight:700;color:\${color};">\${pct}%</div>
                        <div style="font-size:0.8em;color:#6b7280;margin-top:1px;">\${label}</div>
                    </div>\`;
                }

                // ── Pipeline scorecard ──────────────────────────────────────
                messageHTML += \`
                <div style="margin-top:12px;border:1px solid #e5e7eb;border-radius:8px;overflow:hidden;font-size:0.85em;">
                    <div style="background:#f8fafc;padding:8px 12px;border-bottom:1px solid #e5e7eb;font-weight:600;color:#374151;display:flex;justify-content:space-between;">
                        <span>📊 Pipeline Scorecard</span>
                        <span style="color:#6b7280;font-weight:400;">⚡ \${procTime}ms</span>
                    </div>
                    <div style="padding:10px 12px;">
                        <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:10px;">
                            <div>
                                <div style="color:#6b7280;margin-bottom:3px;">Top similarity</div>
                                \${scoreBar(topScore)}
                            </div>
                            <div>
                                <div style="color:#6b7280;margin-bottom:3px;">Avg similarity</div>
                                \${scoreBar(avgScore)}
                            </div>
                        </div>
                        <div style="border-top:1px solid #f3f4f6;padding-top:8px;margin-bottom:2px;">
                            <div style="color:#6b7280;font-size:0.85em;margin-bottom:6px;">Retrieval metrics <span style="font-weight:400;color:#9ca3af;">(threshold ≥ 50% similarity)</span></div>
                            <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:6px;">
                                <div style="background:#f8fafc;border:1px solid #e5e7eb;border-radius:6px;padding:6px 10px;text-align:center;">
                                    <div style="font-size:1.1em;font-weight:700;color:#1f2937;">\${chunks}</div>
                                    <div style="font-size:0.8em;color:#6b7280;margin-top:1px;">Chunks</div>
                                </div>
                                \${metricBadge('Precision@K', precisionAtK)}
                                \${metricBadge('MRR', mrr)}
                                \${metricBadge('NDCG', ndcg)}
                            </div>
                            <details style="margin-top:8px;">
                                <summary style="cursor:pointer;font-size:0.8em;color:#9ca3af;user-select:none;">What do these mean?</summary>
                                <div style="margin-top:6px;background:#f8fafc;border-radius:6px;padding:8px 10px;font-size:0.8em;color:#4b5563;line-height:1.6;">
                                    <div style="margin-bottom:4px;"><strong style="color:#374151;">Precision@K</strong> — of the \${chunks} chunks retrieved, how many are actually relevant (≥50% similarity). <em>\${relevantChunks.length} of \${chunks} = \${Math.round(precisionAtK*100)}%.</em> Assumes similarity ≥ 50% is a proxy for relevance — no ground-truth labels used.</div>
                                    <div style="margin-bottom:4px;"><strong style="color:#374151;">MRR</strong> — Mean Reciprocal Rank. Was the first relevant chunk ranked #1? Score = 1/rank of first relevant chunk. 100% = top result is relevant, 50% = relevant chunk is at rank 2. Same relevance assumption as above.</div>
                                    <div style="margin-bottom:4px;"><strong style="color:#374151;">NDCG</strong> — Normalised Discounted Cumulative Gain. Rewards putting the most relevant chunks highest. Compares actual ranking to an ideal ranking where all relevant chunks come first. 100% = perfect ordering.</div>
                                    <div style="color:#9ca3af;font-size:0.9em;margin-top:4px;">⚠ These are approximations — without labelled ground truth, "relevant" is defined as cosine similarity ≥ 0.5 using the local all-MiniLM-L6-v2 model (384-dim). For exact Precision/Recall, see the Evaluate page.</div>
                                </div>
                            </details>
                        </div>
                        <div style="border-top:1px solid #f3f4f6;padding-top:8px;margin-top:8px;display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;">
                            <div>
                                <div style="color:#6b7280;margin-bottom:3px;">Groundedness</div>
                                <div style="color:#9ca3af;font-style:italic;font-size:0.9em;">TruLens pending</div>
                            </div>
                            <div>
                                <div style="color:#6b7280;margin-bottom:3px;">Answer relevance</div>
                                <div style="color:#9ca3af;font-style:italic;font-size:0.9em;">TruLens pending</div>
                            </div>
                            <div>
                                <div style="color:#6b7280;margin-bottom:3px;">Context relevance</div>
                                <div style="color:#9ca3af;font-style:italic;font-size:0.9em;">TruLens pending</div>
                            </div>
                        </div>
                    </div>

                    \${chunks > 0 ? \`
                    <details style="border-top:1px solid #e5e7eb;">
                        <summary style="padding:8px 12px;cursor:pointer;color:#374151;font-weight:600;background:#f8fafc;">
                            📄 Retrieved chunks (\${chunks})
                        </summary>
                        <div style="padding:8px 12px;">
                        \${sources.map((s, i) => \`
                            <div style="margin:6px 0;padding:8px;background:#f9fafb;border-radius:6px;border-left:3px solid \${scoreColor(s.similarity)};">
                                <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                                    <span style="font-weight:600;">Chunk \${s.chunkIndex}\${s.section ? ' <span style="font-weight:400;color:#6b7280;">· ' + s.section + '</span>' : ''}</span>
                                    <span style="color:\${scoreColor(s.similarity)};font-weight:600;">\${(s.similarity*100).toFixed(1)}% hybrid</span>
                                </div>
                                \${scoreBar(s.similarity)}
                                \${s.cosineScore !== undefined ? \`
                                <div style="display:flex;gap:12px;margin-top:4px;font-size:0.8em;color:#6b7280;">
                                    <span>Cosine: <strong>\${(s.cosineScore*100).toFixed(1)}%</strong></span>
                                    <span>BM25: <strong>\${s.bm25Score !== undefined ? s.bm25Score.toFixed(3) : 'n/a'}</strong></span>
                                </div>\` : ''}
                                <div style="margin-top:6px;color:#4b5563;font-size:0.9em;font-style:italic;">"\${s.preview.substring(0,120)}..."</div>
                            </div>
                        \`).join('')}
                        </div>
                    </details>
                    \` : ''}
                </div>\`;
            }

            if (metadata.error) {
                messageHTML = \`<div class="message-content error">❌ \${content}</div>\`;
            }

            messageDiv.innerHTML = messageHTML;
            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;

            return messageId;
        }

        function updateSendButton(enabled) {
            const sendBtn = document.getElementById('sendBtn');
            sendBtn.disabled = !enabled;
            sendBtn.textContent = enabled ? 'Send' : 'Sending...';
        }

        function handleKeyPress(event) {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                sendMessage();
            }
        }

        function setMessage(message) {
            document.getElementById('messageInput').value = message;
            document.getElementById('messageInput').focus();
        }

        async function clearChat() {
            if (!confirm('Clear conversation history?')) return;

            try {
                await fetch('/api/rag-docs/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        message: 'clear',
                        clearHistory: true
                    })
                });

                // Clear UI messages except welcome message
                const messagesDiv = document.getElementById('chatMessages');
                const messages = messagesDiv.querySelectorAll('.message');
                for (let i = 1; i < messages.length; i++) {
                    messages[i].remove();
                }

            } catch (error) {
                console.error('Error clearing chat:', error);
            }
        }

        // Focus input on load
        window.addEventListener('load', () => {
            document.getElementById('messageInput').focus();
        });
    </script>
</body>
</html>`;

  res.set('Content-Type', 'text/html');
  res.send(html);
});

/**
 * POST /api/rag-docs/chat/clear - Clear conversation history
 */
router.post('/chat/clear', async (req, res) => {
  try {
    const chatService = await getCachedChatService();
    chatService.clearHistory();

    res.json({
      success: true,
      message: 'Conversation history cleared'
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

/**
 * GET /api/rag-docs/chat/stats - Get conversation statistics
 */
router.get('/chat/stats', async (req, res) => {
  try {
    const chatService = await getCachedChatService();
    const stats = chatService.getConversationStats();

    res.json({
      success: true,
      stats: stats,
      documents: chatService.getDocuments()
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

/**
 * POST /api/rag-docs/compare - Compare different retrieval configurations
 */
router.post('/compare', async (req, res) => {
  try {
    const chatService = await getCachedChatService();
    const { configurations, customQueries } = req.body;

    console.log('🔬 Running configuration comparison...');

    const comparison = await chatService.compareRetrievalConfigs(configurations, customQueries);

    res.json({
      success: true,
      comparison: comparison,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    console.error('❌ Comparison error:', error.message);
    res.status(500).json({
      success: false,
      error: error.message,
      helpText: 'Make sure configurations are properly formatted'
    });
  }
});

module.exports = router;