/**
 * RetrievalEvaluator - Measures accuracy and quality of RAG retrieval
 */
class RetrievalEvaluator {
  constructor(vectorStore) {
    this.vectorStore = vectorStore;
  }

  /**
   * Evaluate retrieval accuracy for test queries
   * @param {Array} testQueries - Array of {question, expectedSources, expectedInfo, difficulty}
   * @param {Object} options - Evaluation options
   * @returns {Object} Evaluation metrics
   */
  async evaluateRetrieval(testQueries, options = {}) {
    const { topK = 5, threshold = 0.1 } = options;
    const results = [];

    console.log(`📊 Evaluating ${testQueries.length} test queries...`);

    for (const testQuery of testQueries) {
      const startTime = Date.now();

      const retrievedResults = await this.vectorStore.similaritySearch(
        testQuery.question,
        topK,
        threshold
      );

      const retrievalTime = Date.now() - startTime;

      // Calculate metrics
      const relevantRetrieved = retrievedResults.filter(result =>
        testQuery.expectedSources.includes(result.metadata.filename)
      );

      // Document-level recall (more meaningful for RAG)
      const uniqueRetrievedDocs = [...new Set(retrievedResults.map(r => r.metadata.filename))];
      const relevantDocsFound = uniqueRetrievedDocs.filter(doc =>
        testQuery.expectedSources.includes(doc)
      );

      console.log(`🔍 Query: "${testQuery.question}"`);
      console.log(`   Expected sources: ${JSON.stringify(testQuery.expectedSources)}`);
      console.log(`   Retrieved count: ${retrievedResults.length}`);
      console.log(`   Relevant chunks: ${relevantRetrieved.length}`);
      console.log(`   Relevant docs found: ${relevantDocsFound.length}`);
      console.log(`   Retrieved sources: ${retrievedResults.map(r => r.metadata.filename)}`);

      // Chunk-level precision (what % of retrieved chunks were relevant)
      const precision = relevantRetrieved.length / Math.max(retrievedResults.length, 1);

      // Document-level recall (did we find the right documents?)
      const recall = relevantDocsFound.length / testQuery.expectedSources.length;

      // Content coverage (how much of the relevant content did we retrieve?)
      const contentCoverage = relevantRetrieved.length / Math.max(relevantRetrieved.length, 1);
      const f1Score = precision + recall > 0 ? 2 * (precision * recall) / (precision + recall) : 0;

      // Mean Reciprocal Rank (MRR) - position of first relevant result
      let mrr = 0;
      for (let i = 0; i < retrievedResults.length; i++) {
        if (testQuery.expectedSources.includes(retrievedResults[i].metadata.filename)) {
          mrr = 1 / (i + 1);
          break;
        }
      }

      // NDCG@K approximation
      const ndcg = this.calculateNDCG(retrievedResults, testQuery.expectedSources, topK);

      results.push({
        question: testQuery.question,
        difficulty: testQuery.difficulty || 'medium',
        retrievedCount: retrievedResults.length,
        relevantCount: relevantRetrieved.length,
        expectedCount: testQuery.expectedSources.length,
        precision: Math.round(precision * 100) / 100,
        recall: Math.round(recall * 100) / 100,
        f1Score: Math.round(f1Score * 100) / 100,
        mrr: Math.round(mrr * 100) / 100,
        ndcg: Math.round(ndcg * 100) / 100,
        retrievalTime,
        topResult: retrievedResults[0] || null,
        retrievedSources: retrievedResults.map(r => ({
          filename: r.metadata.filename,
          similarity: r.similarity,
          chunkIndex: r.metadata.chunkIndex,
          content: r.content
        }))
      });
    }

    return this.calculateAggregateMetrics(results);
  }

  /**
   * Calculate NDCG@K (Normalized Discounted Cumulative Gain)
   */
  calculateNDCG(retrievedResults, expectedSources, k) {
    if (retrievedResults.length === 0) return 0;

    // DCG calculation
    let dcg = 0;
    for (let i = 0; i < Math.min(retrievedResults.length, k); i++) {
      const isRelevant = expectedSources.includes(retrievedResults[i].metadata.filename) ? 1 : 0;
      dcg += isRelevant / Math.log2(i + 2);
    }

    // IDCG calculation (ideal case)
    let idcg = 0;
    const relevantCount = Math.min(expectedSources.length, k);
    for (let i = 0; i < relevantCount; i++) {
      idcg += 1 / Math.log2(i + 2);
    }

    return idcg > 0 ? dcg / idcg : 0;
  }

  /**
   * Calculate aggregate metrics across all queries
   */
  calculateAggregateMetrics(results) {
    const avgPrecision = results.reduce((sum, r) => sum + r.precision, 0) / results.length;
    const avgRecall = results.reduce((sum, r) => sum + r.recall, 0) / results.length;
    const avgF1 = results.reduce((sum, r) => sum + r.f1Score, 0) / results.length;
    const avgMRR = results.reduce((sum, r) => sum + r.mrr, 0) / results.length;
    const avgNDCG = results.reduce((sum, r) => sum + r.ndcg, 0) / results.length;
    const avgRetrievalTime = results.reduce((sum, r) => sum + r.retrievalTime, 0) / results.length;

    return {
      overall: {
        avgPrecision: Math.round(avgPrecision * 100) / 100,
        avgRecall: Math.round(avgRecall * 100) / 100,
        avgF1: Math.round(avgF1 * 100) / 100,
        avgMRR: Math.round(avgMRR * 100) / 100,
        avgNDCG: Math.round(avgNDCG * 100) / 100,
        avgRetrievalTime: Math.round(avgRetrievalTime),
        totalQueries: results.length
      },
      byDifficulty: this.groupResultsByDifficulty(results),
      detailed: results
    };
  }

  /**
   * Group evaluation results by difficulty level
   */
  groupResultsByDifficulty(results) {
    const grouped = {};

    for (const result of results) {
      const diff = result.difficulty;
      if (!grouped[diff]) {
        grouped[diff] = {
          count: 0,
          avgPrecision: 0,
          avgRecall: 0,
          avgF1: 0,
          avgMRR: 0,
          avgNDCG: 0,
          avgRetrievalTime: 0
        };
      }
      grouped[diff].count++;
      grouped[diff].avgPrecision += result.precision;
      grouped[diff].avgRecall += result.recall;
      grouped[diff].avgF1 += result.f1Score;
      grouped[diff].avgMRR += result.mrr;
      grouped[diff].avgNDCG += result.ndcg;
      grouped[diff].avgRetrievalTime += result.retrievalTime;
    }

    // Calculate averages
    for (const diff in grouped) {
      const count = grouped[diff].count;
      grouped[diff].avgPrecision = Math.round((grouped[diff].avgPrecision / count) * 100) / 100;
      grouped[diff].avgRecall = Math.round((grouped[diff].avgRecall / count) * 100) / 100;
      grouped[diff].avgF1 = Math.round((grouped[diff].avgF1 / count) * 100) / 100;
      grouped[diff].avgMRR = Math.round((grouped[diff].avgMRR / count) * 100) / 100;
      grouped[diff].avgNDCG = Math.round((grouped[diff].avgNDCG / count) * 100) / 100;
      grouped[diff].avgRetrievalTime = Math.round(grouped[diff].avgRetrievalTime / count);
    }

    return grouped;
  }

  /**
   * Create test queries for evaluation
   * @param {Array} documents - Document metadata
   * @returns {Array} Generated test queries
   */
  static createTestQueries(documents) {
    const testQueries = [];

    console.log(`🔍 Creating test queries for ${documents.length} documents`);

    if (documents.length === 0) {
      console.warn('⚠️  No documents provided to createTestQueries');
      return [];
    }

    // Generate queries for each document
    documents.forEach((doc, index) => {
      // Handle both doc.metadata.filename and doc.filename formats
      const filename = doc.metadata?.filename || doc.filename || `document_${index}`;
      const pages = doc.metadata?.pages || doc.pages || 1;

      console.log(`📄 Creating queries for: ${filename} (${pages} pages)`);

      // Easy queries (general document questions)
      testQueries.push({
        question: `What is the main topic of ${filename}?`,
        expectedSources: [filename],
        difficulty: 'easy',
        type: 'general'
      });

      // Medium queries (specific content)
      if (pages > 5) {
        testQueries.push({
          question: `What information is covered in the middle sections of ${filename}?`,
          expectedSources: [filename],
          difficulty: 'medium',
          type: 'middle_content'
        });
      }

      // Hard queries (later pages/sections)
      if (pages > 10) {
        testQueries.push({
          question: `What conclusions or final information is provided in ${filename}?`,
          expectedSources: [filename],
          difficulty: 'hard',
          type: 'late_content'
        });
      }
    });

    // Cross-document queries
    if (documents.length > 1) {
      testQueries.push({
        question: "What are the common themes across all documents?",
        expectedSources: documents.map(d => d.metadata?.filename || d.filename),
        difficulty: 'hard',
        type: 'cross_document'
      });
    }

    console.log(`✅ Generated ${testQueries.length} test queries`);

    if (testQueries.length === 0) {
      console.warn('⚠️  No test queries generated - this will cause evaluation to fail');
    }

    return testQueries;
  }

  /**
   * Run a comparison between two configurations
   */
  async compareConfigurations(testQueries, configs) {
    const results = {};

    for (const configName in configs) {
      console.log(`\n🔧 Testing configuration: ${configName}`);
      const config = configs[configName];

      // Apply configuration changes here
      // For example: vectorStore.setThreshold(config.threshold)

      results[configName] = await this.evaluateRetrieval(testQueries, config);
    }

    return results;
  }

  /**
   * Generate evaluation report
   */
  generateReport(evaluation, configName = 'Current') {
    console.log(`\n📊 RAG Retrieval Evaluation Report - ${configName}`);
    console.log('='.repeat(50));

    const overall = evaluation.overall;
    console.log(`📈 Overall Metrics:`);
    console.log(`   Precision: ${overall.avgPrecision} (higher is better)`);
    console.log(`   Recall: ${overall.avgRecall} (higher is better)`);
    console.log(`   F1-Score: ${overall.avgF1} (higher is better)`);
    console.log(`   MRR: ${overall.avgMRR} (higher is better)`);
    console.log(`   NDCG: ${overall.avgNDCG} (higher is better)`);
    console.log(`   Avg Retrieval Time: ${overall.avgRetrievalTime}ms`);
    console.log(`   Total Queries: ${overall.totalQueries}`);

    console.log(`\n📊 By Difficulty:`);
    for (const difficulty in evaluation.byDifficulty) {
      const metrics = evaluation.byDifficulty[difficulty];
      console.log(`   ${difficulty.toUpperCase()}: F1=${metrics.avgF1}, MRR=${metrics.avgMRR}, Count=${metrics.count}`);
    }

    return evaluation;
  }
}

module.exports = RetrievalEvaluator;