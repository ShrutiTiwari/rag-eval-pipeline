const OpenAI = require('openai');
const DocumentLoader = require('./DocumentLoader');
const VectorStore = require('./VectorStore');
const RetrievalEvaluator = require('./RetrievalEvaluator');
const TruLensClient = require('./TruLensClient');

/**
 * ChatService - RAG pipeline orchestrator
 * Combines document retrieval with LLM generation for contextual answers
 */
class ChatService {
  constructor(apiKey = null) {
    this.openai = new OpenAI({
      apiKey: apiKey || process.env.OPENAI_API_KEY
    });
    this.vectorStore = null;
    this.documents = [];
    this.chatModel = 'gpt-4o-mini'; // Cost-effective chat model
    this.conversationHistory = [];
    this.maxContextLength = 8000; // Token limit for context
    this.maxHistoryMessages = 10; // Limit conversation history
    this.evaluator = null; // Will be initialized with vector store
    this.truLensClient = new TruLensClient(); // TruLens evaluation client
  }

  /**
   * Initialize the chat service with documents
   * @param {string} documentsPath - Path to PDF documents directory
   * @param {Object} options - Configuration options
   */
  async initialize(documentsPath, options = {}) {
    const {
      chunkSize = 1000,
      chunkOverlap = 100,
      maxDocuments = 3, // Limit for cost control
      maxChunksPerDoc = 10 // Limit chunks per document
    } = options;

    console.log('🚀 Initializing ChatService with RAG pipeline...');

    // Load documents
    console.log('📁 Loading documents...');
    this.documents = await DocumentLoader.loadDocuments(documentsPath);

    if (this.documents.length === 0) {
      throw new Error('No documents found for ChatService initialization');
    }

    console.log(`📄 Loaded ${this.documents.length} documents`);

    // Initialize vector store
    console.log('🔮 Initializing vector store...');
    this.vectorStore = new VectorStore();

    // Process documents and generate embeddings
    const limitedDocuments = this.documents.slice(0, maxDocuments);
    for (const document of limitedDocuments) {
      console.log(`⚡ Processing: ${document.metadata.filename}`);

      const chunks = await DocumentLoader.chunkDocument(
        document.content,
        chunkSize,
        chunkOverlap
      );

      const limitedChunks = chunks.slice(0, maxChunksPerDoc);
      console.log(`   Chunks: ${limitedChunks.length}/${chunks.length}`);

      const embeddings = await this.vectorStore.generateEmbeddings(
        limitedChunks,
        document.metadata
      );
      this.vectorStore.addEmbeddings(embeddings);
    }

    const stats = this.vectorStore.getStats();
    console.log(`✅ ChatService initialized with ${stats.totalEmbeddings} embeddings from ${stats.uniqueDocuments} documents`);

    // Initialize evaluator
    this.evaluator = new RetrievalEvaluator(this.vectorStore);
  }

  /**
   * Generate a chat response using RAG pipeline
   * @param {string} userMessage - User's question or message
   * @param {Object} options - Configuration options
   * @returns {Promise<Object>} Chat response with sources
   */
  async chat(userMessage, options = {}) {
    const {
      topK = 3,
      similarityThreshold = 0.1,
      includeHistory = true,
      temperature = 0.7,
      maxTokens = 1000
    } = options;

    if (!this.vectorStore) {
      throw new Error('ChatService not initialized. Call initialize() first.');
    }

    console.log(`💬 Processing chat request: "${userMessage}"`);

    try {
      // Step 1: Retrieve relevant context
      const retrievalResults = await this.vectorStore.similaritySearch(
        userMessage,
        topK,
        similarityThreshold
      );

      console.log(`🔍 Retrieved ${retrievalResults.length} relevant chunks`);

      if (retrievalResults.length === 0) {
        return this.generateFallbackResponse(userMessage);
      }

      // Step 2: Assemble context from retrieved chunks
      const context = this.assembleContext(retrievalResults);

      // Step 3: Build conversation messages
      const messages = this.buildMessages(userMessage, context, includeHistory);

      // Step 4: Generate LLM response
      const response = await this.openai.chat.completions.create({
        model: this.chatModel,
        messages: messages,
        temperature: temperature,
        max_tokens: maxTokens,
        presence_penalty: 0.1,
        frequency_penalty: 0.1
      });

      const assistantMessage = response.choices[0].message.content;

      // Step 5: Update conversation history
      if (includeHistory) {
        this.addToHistory('user', userMessage);
        this.addToHistory('assistant', assistantMessage);
      }

      // Step 6: Format response with metadata
      return {
        success: true,
        response: assistantMessage,
        sources: this.formatSources(retrievalResults),
        metadata: {
          retrievedChunks: retrievalResults.length,
          model: this.chatModel,
          temperature: temperature,
          tokensUsed: response.usage?.total_tokens || 0,
          processingTime: Date.now() - Date.now() // Will be calculated properly
        },
        conversationLength: this.conversationHistory.length
      };

    } catch (error) {
      console.error('❌ Chat generation error:', error.message);
      throw new Error(`Failed to generate chat response: ${error.message}`);
    }
  }

  /**
   * Assemble context from retrieved chunks
   * @param {Array} retrievalResults - Results from vector search
   * @returns {string} Assembled context text
   */
  assembleContext(retrievalResults) {
    const contextParts = retrievalResults.map((result, index) => {
      return `[Source ${index + 1}: ${result.metadata.filename}, chunk ${result.metadata.chunkIndex}]\n${result.content}\n`;
    });

    const fullContext = contextParts.join('\n---\n\n');

    // Truncate if too long (rough token estimation: 1 token ≈ 4 chars)
    if (fullContext.length > this.maxContextLength * 4) {
      console.log('⚠️ Context truncated due to length limits');
      return fullContext.substring(0, this.maxContextLength * 4) + '\n\n[Context truncated...]';
    }

    return fullContext;
  }

  /**
   * Build conversation messages for OpenAI
   * @param {string} userMessage - Current user message
   * @param {string} context - Retrieved context
   * @param {boolean} includeHistory - Whether to include conversation history
   * @returns {Array} Messages array for OpenAI
   */
  buildMessages(userMessage, context, includeHistory) {
    const messages = [];

    // System prompt
    messages.push({
      role: 'system',
      content: `You are a helpful assistant that answers questions based on the provided documents.

Guidelines:
1. Answer based ONLY on the information provided in the context
2. If the answer is not in the context, say "I don't have enough information to answer that question based on the provided documents"
3. Be specific and cite relevant details from the sources
4. If multiple sources provide relevant information, synthesize them coherently
5. Keep responses clear and well-structured
6. When referencing information, mention the source document when helpful

Context from relevant documents:
${context}`
    });

    // Add conversation history if enabled
    if (includeHistory && this.conversationHistory.length > 0) {
      const recentHistory = this.conversationHistory.slice(-this.maxHistoryMessages);
      messages.push(...recentHistory);
    }

    // Current user message
    messages.push({
      role: 'user',
      content: userMessage
    });

    return messages;
  }

  /**
   * Generate fallback response when no relevant context is found
   * @param {string} userMessage - User's message
   * @returns {Object} Fallback response object
   */
  generateFallbackResponse(userMessage) {
    const suggestions = [
      "Try asking about piano scales, ABRSM requirements, or practice techniques",
      "Check if your question relates to music education or exam syllabuses",
      "Try using different keywords or phrasing your question differently"
    ];

    return {
      success: true,
      response: `I don't have enough information to answer that question based on the available documents. Here are some suggestions:\n\n${suggestions.map(s => `• ${s}`).join('\n')}`,
      sources: [],
      metadata: {
        retrievedChunks: 0,
        fallbackResponse: true,
        suggestions: suggestions
      },
      conversationLength: this.conversationHistory.length
    };
  }

  /**
   * Format sources for response metadata
   * @param {Array} retrievalResults - Vector search results
   * @returns {Array} Formatted source information
   */
  formatSources(retrievalResults) {
    return retrievalResults.map((result, index) => ({
      id: index + 1,
      filename: result.metadata.filename,
      chunkIndex: result.metadata.chunkIndex,
      similarity: Math.round(result.similarity * 100) / 100,
      preview: result.content.substring(0, 150) + '...',
      position: {
        startChar: result.metadata.startChar,
        endChar: result.metadata.endChar
      }
    }));
  }

  /**
   * Add message to conversation history
   * @param {string} role - 'user' or 'assistant'
   * @param {string} content - Message content
   */
  addToHistory(role, content) {
    this.conversationHistory.push({ role, content });

    // Trim history if too long
    if (this.conversationHistory.length > this.maxHistoryMessages * 2) {
      this.conversationHistory = this.conversationHistory.slice(-this.maxHistoryMessages * 2);
    }
  }

  /**
   * Clear conversation history
   */
  clearHistory() {
    this.conversationHistory = [];
    console.log('🗑️ Conversation history cleared');
  }

  /**
   * Get conversation statistics
   * @returns {Object} Statistics about the current conversation
   */
  getConversationStats() {
    const userMessages = this.conversationHistory.filter(msg => msg.role === 'user');
    const assistantMessages = this.conversationHistory.filter(msg => msg.role === 'assistant');

    return {
      totalMessages: this.conversationHistory.length,
      userMessages: userMessages.length,
      assistantMessages: assistantMessages.length,
      vectorStoreStats: this.vectorStore ? this.vectorStore.getStats() : null,
      documentsLoaded: this.documents.length,
      chatModel: this.chatModel
    };
  }

  /**
   * Get available documents information
   * @returns {Array} Document metadata
   */
  getDocuments() {
    return this.documents.map(doc => ({
      filename: doc.metadata.filename,
      pages: doc.metadata.pages,
      characters: doc.content.length,
      extractedAt: doc.metadata.extractedAt
    }));
  }

  /**
   * Test the RAG pipeline with a sample query
   * @param {string} testQuery - Query to test with
   * @returns {Promise<Object>} Test results
   */
  async testRAGPipeline(testQuery = "What are the main requirements?") {
    console.log(`🧪 Testing RAG pipeline with query: "${testQuery}"`);

    const startTime = Date.now();

    try {
      const response = await this.chat(testQuery, {
        topK: 2,
        temperature: 0.3,
        includeHistory: false
      });

      const endTime = Date.now();
      const processingTime = endTime - startTime;

      return {
        success: true,
        testQuery,
        response: response.response,
        processingTime,
        sourcesCount: response.sources.length,
        tokensUsed: response.metadata.tokensUsed,
        pipelineWorking: true
      };

    } catch (error) {
      return {
        success: false,
        testQuery,
        error: error.message,
        processingTime: Date.now() - startTime,
        pipelineWorking: false
      };
    }
  }

  /**
   * Evaluate retrieval accuracy with current configuration
   * @param {Array} customQueries - Optional custom test queries
   * @returns {Promise<Object>} Evaluation results
   */
  async evaluateRetrieval(customQueries = null) {
    if (!this.evaluator) {
      throw new Error('Evaluator not initialized. Call initialize() first.');
    }

    console.log(`🔍 Debug: documents array length: ${this.documents.length}`);
    console.log(`🔍 Debug: documents structure:`, this.documents.map(d => ({
      filename: d.metadata?.filename,
      pages: d.metadata?.pages,
      contentLength: d.content?.length
    })));

    // Use custom queries or generate default ones
    const testQueries = customQueries || RetrievalEvaluator.createTestQueries(this.documents);

    console.log(`📊 Running retrieval evaluation with ${testQueries.length} queries...`);
    console.log(`🔍 Debug: test queries:`, testQueries.map(q => ({
      question: q.question,
      expectedSources: q.expectedSources
    })));

    if (testQueries.length === 0) {
      throw new Error('No test queries generated - check document structure');
    }

    const evaluation = await this.evaluator.evaluateRetrieval(testQueries);
    this.evaluator.generateReport(evaluation);

    return evaluation;
  }

  /**
   * Compare retrieval accuracy between configurations
   * @param {Object} configurations - Different config settings to compare
   * @param {Array} customQueries - Optional custom test queries
   * @returns {Promise<Object>} Comparison results
   */
  async compareRetrievalConfigs(configurations, customQueries = null) {
    if (!this.evaluator) {
      throw new Error('Evaluator not initialized. Call initialize() first.');
    }

    const testQueries = customQueries || RetrievalEvaluator.createTestQueries(this.documents);

    console.log(`\n🔬 Comparing ${Object.keys(configurations).length} configurations...`);

    const results = {};

    for (const [configName, config] of Object.entries(configurations)) {
      console.log(`\n📋 Testing: ${configName}`);

      const evaluation = await this.evaluator.evaluateRetrieval(testQueries, config);
      this.evaluator.generateReport(evaluation, configName);

      results[configName] = evaluation;
    }

    // Compare results
    console.log(`\n📊 Configuration Comparison Summary:`);
    console.log('='.repeat(60));

    for (const [configName, result] of Object.entries(results)) {
      const overall = result.overall;
      console.log(`${configName}: F1=${overall.avgF1}, MRR=${overall.avgMRR}, Time=${overall.avgRetrievalTime}ms`);
    }

    return results;
  }

  /**
   * Evaluate a chat response using TruLens
   * @param {string} userMessage - User's question
   * @param {Array} retrievedResults - Retrieved chunks from vector search
   * @param {string} generatedResponse - Generated response
   * @param {Object} metadata - Additional metadata
   * @returns {Promise<Object>} TruLens evaluation results
   */
  async evaluateWithTruLens(userMessage, retrievedResults, generatedResponse, metadata = {}) {
    try {
      // Check if TruLens service is available
      const isAvailable = await this.truLensClient.isAvailable();
      if (!isAvailable) {
        console.warn('⚠️ TruLens service not available, skipping evaluation');
        return null;
      }

      console.log('🔍 Evaluating with TruLens...');

      const evaluation = await this.truLensClient.evaluateSingle(
        userMessage,
        retrievedResults.map(r => r.content),
        generatedResponse,
        {
          ...metadata,
          retrievedCount: retrievedResults.length,
          sources: retrievedResults.map(r => r.metadata.filename),
          model: this.chatModel
        }
      );

      return this.truLensClient.formatResults(evaluation);

    } catch (error) {
      console.error('❌ TruLens evaluation error:', error.message);
      return null;
    }
  }

  /**
   * Enhanced chat method with optional TruLens evaluation
   * @param {string} userMessage - User's question or message
   * @param {Object} options - Configuration options
   * @returns {Promise<Object>} Chat response with optional TruLens evaluation
   */
  async chatWithEvaluation(userMessage, options = {}) {
    const {
      includeTruLensEval = false,
      ...chatOptions
    } = options;

    // Get standard chat response
    const response = await this.chat(userMessage, chatOptions);

    // Add TruLens evaluation if requested
    if (includeTruLensEval && response.success) {
      console.log('🔍 Adding TruLens evaluation...');

      // Reconstruct retrieved results for TruLens
      const retrievedResults = response.sources.map(source => ({
        content: source.preview + '...', // Use preview as content
        metadata: {
          filename: source.filename,
          chunkIndex: source.chunkIndex
        }
      }));

      const truLensEval = await this.evaluateWithTruLens(
        userMessage,
        retrievedResults,
        response.response,
        response.metadata
      );

      response.truLensEvaluation = truLensEval;
    }

    return response;
  }

  /**
   * Run comprehensive evaluation including both custom and TruLens metrics
   * @param {Array} customQueries - Optional custom test queries
   * @returns {Promise<Object>} Comprehensive evaluation results
   */
  async comprehensiveEvaluation(customQueries = null) {
    console.log('🚀 Running comprehensive evaluation with TruLens...');

    // Run standard retrieval evaluation
    const retrievalEval = await this.evaluateRetrieval(customQueries);

    // Check if TruLens is available
    const truLensAvailable = await this.truLensClient.isAvailable();
    if (!truLensAvailable) {
      console.warn('⚠️ TruLens service not available, running basic evaluation only');
      return {
        retrievalMetrics: retrievalEval,
        truLensMetrics: null,
        warning: 'TruLens service not available'
      };
    }

    // Generate sample conversations for TruLens evaluation
    const testQueries = customQueries || RetrievalEvaluator.createTestQueries(this.documents);
    const truLensEvaluations = [];

    console.log(`📊 Evaluating ${testQueries.length} queries with TruLens...`);

    for (const query of testQueries.slice(0, 3)) { // Limit to 3 for cost control
      try {
        console.log(`   Evaluating: "${query.question}"`);

        // Get chat response
        const response = await this.chat(query.question, {
          includeHistory: false,
          temperature: 0.7
        });

        if (response.success) {
          // Reconstruct retrieved results
          const retrievedResults = response.sources.map(source => ({
            content: source.preview + '...', // Limited content
            metadata: {
              filename: source.filename,
              chunkIndex: source.chunkIndex
            }
          }));

          const truLensEval = await this.evaluateWithTruLens(
            query.question,
            retrievedResults,
            response.response,
            {
              difficulty: query.difficulty,
              type: query.type
            }
          );

          if (truLensEval) {
            truLensEvaluations.push({
              query: query.question,
              difficulty: query.difficulty,
              evaluation: truLensEval
            });
          }
        }

      } catch (error) {
        console.error(`❌ Error evaluating query "${query.question}":`, error.message);
      }
    }

    // Aggregate TruLens results
    let truLensMetrics = null;
    if (truLensEvaluations.length > 0) {
      const scores = truLensEvaluations.map(e => e.evaluation.scores);
      const metrics = ['groundedness', 'answer_relevance', 'context_relevance', 'coherence', 'retrieval_precision'];

      const aggregated = {};
      for (const metric of metrics) {
        const metricScores = scores.map(s => s[metric]).filter(s => s !== undefined && s > 0);
        if (metricScores.length > 0) {
          aggregated[metric] = {
            average: metricScores.reduce((sum, score) => sum + score, 0) / metricScores.length,
            min: Math.min(...metricScores),
            max: Math.max(...metricScores),
            count: metricScores.length
          };
        }
      }

      const overallScores = truLensEvaluations.map(e => e.evaluation.overallScore).filter(s => s > 0);
      aggregated.overall = {
        average: overallScores.length > 0 ? overallScores.reduce((sum, score) => sum + score, 0) / overallScores.length : 0,
        min: overallScores.length > 0 ? Math.min(...overallScores) : 0,
        max: overallScores.length > 0 ? Math.max(...overallScores) : 0,
        count: overallScores.length
      };

      truLensMetrics = {
        aggregated: aggregated,
        individual: truLensEvaluations,
        totalEvaluations: truLensEvaluations.length
      };
    }

    return {
      retrievalMetrics: retrievalEval,
      truLensMetrics: truLensMetrics,
      timestamp: new Date().toISOString()
    };
  }
}

module.exports = ChatService;