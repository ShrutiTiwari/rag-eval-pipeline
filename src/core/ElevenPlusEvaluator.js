/**
 * ElevenPlusEvaluator - Retrieval evaluator specific to 11-plus-syllabus.pdf
 * Use this instead of RetrievalEvaluator when testing against the 11+ document.
 * RetrievalEvaluator.js remains intact for ABRSM and other multi-doc testing.
 */
class ElevenPlusEvaluator {
  constructor(vectorStore) {
    this.vectorStore = vectorStore;
    this.targetDoc = '11-plus-syllabus.pdf';
  }

  /**
   * Evaluate retrieval accuracy for test queries
   * @param {Array} testQueries - Array of {question, expectedSources, difficulty}
   * @param {Object} options - Evaluation options
   * @returns {Object} Evaluation metrics
   */
  async evaluateRetrieval(testQueries, options = {}) {
    const { topK = 5, threshold = 0.1 } = options;
    const results = [];

    console.log(`📊 Evaluating ${testQueries.length} test queries against ${this.targetDoc}...`);

    for (const testQuery of testQueries) {
      const startTime = Date.now();

      const retrievedResults = await this.vectorStore.similaritySearch(
        testQuery.question,
        topK,
        threshold
      );

      const retrievalTime = Date.now() - startTime;

      const relevantRetrieved = retrievedResults.filter(result =>
        testQuery.expectedSources.includes(result.metadata.filename)
      );

      const uniqueRetrievedDocs = [...new Set(retrievedResults.map(r => r.metadata.filename))];
      const relevantDocsFound = uniqueRetrievedDocs.filter(doc =>
        testQuery.expectedSources.includes(doc)
      );

      console.log(`🔍 Query: "${testQuery.question}"`);
      console.log(`   Expected sources: ${JSON.stringify(testQuery.expectedSources)}`);
      console.log(`   Retrieved count: ${retrievedResults.length}`);
      console.log(`   Relevant chunks: ${relevantRetrieved.length}`);
      console.log(`   Relevant docs found: ${relevantDocsFound.length}`);

      const precision = relevantRetrieved.length / Math.max(retrievedResults.length, 1);
      const recall = relevantDocsFound.length / testQuery.expectedSources.length;
      const f1Score = precision + recall > 0 ? 2 * (precision * recall) / (precision + recall) : 0;

      let mrr = 0;
      for (let i = 0; i < retrievedResults.length; i++) {
        if (testQuery.expectedSources.includes(retrievedResults[i].metadata.filename)) {
          mrr = 1 / (i + 1);
          break;
        }
      }

      const ndcg = this.calculateNDCG(retrievedResults, testQuery.expectedSources, topK);

      results.push({
        question: testQuery.question,
        difficulty: testQuery.difficulty || 'medium',
        type: testQuery.type || 'factual',
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
   * Calculate NDCG@K
   */
  calculateNDCG(retrievedResults, expectedSources, k) {
    if (retrievedResults.length === 0) return 0;

    let dcg = 0;
    for (let i = 0; i < Math.min(retrievedResults.length, k); i++) {
      const isRelevant = expectedSources.includes(retrievedResults[i].metadata.filename) ? 1 : 0;
      dcg += isRelevant / Math.log2(i + 2);
    }

    let idcg = 0;
    const relevantCount = Math.min(expectedSources.length, k);
    for (let i = 0; i < relevantCount; i++) {
      idcg += 1 / Math.log2(i + 2);
    }

    return idcg > 0 ? dcg / idcg : 0;
  }

  /**
   * Aggregate metrics across all queries
   */
  calculateAggregateMetrics(results) {
    const avg = key => results.reduce((sum, r) => sum + r[key], 0) / results.length;

    return {
      overall: {
        avgPrecision: Math.round(avg('precision') * 100) / 100,
        avgRecall: Math.round(avg('recall') * 100) / 100,
        avgF1: Math.round(avg('f1Score') * 100) / 100,
        avgMRR: Math.round(avg('mrr') * 100) / 100,
        avgNDCG: Math.round(avg('ndcg') * 100) / 100,
        avgRetrievalTime: Math.round(avg('retrievalTime')),
        totalQueries: results.length
      },
      byDifficulty: this.groupByDifficulty(results),
      byType: this.groupByType(results),
      detailed: results
    };
  }

  /**
   * Group results by difficulty
   */
  groupByDifficulty(results) {
    const grouped = {};
    for (const result of results) {
      const diff = result.difficulty;
      if (!grouped[diff]) grouped[diff] = { count: 0, avgPrecision: 0, avgRecall: 0, avgF1: 0, avgMRR: 0, avgNDCG: 0 };
      grouped[diff].count++;
      grouped[diff].avgPrecision += result.precision;
      grouped[diff].avgRecall += result.recall;
      grouped[diff].avgF1 += result.f1Score;
      grouped[diff].avgMRR += result.mrr;
      grouped[diff].avgNDCG += result.ndcg;
    }
    for (const diff in grouped) {
      const c = grouped[diff].count;
      grouped[diff].avgPrecision = Math.round(grouped[diff].avgPrecision / c * 100) / 100;
      grouped[diff].avgRecall    = Math.round(grouped[diff].avgRecall    / c * 100) / 100;
      grouped[diff].avgF1        = Math.round(grouped[diff].avgF1        / c * 100) / 100;
      grouped[diff].avgMRR       = Math.round(grouped[diff].avgMRR       / c * 100) / 100;
      grouped[diff].avgNDCG      = Math.round(grouped[diff].avgNDCG      / c * 100) / 100;
    }
    return grouped;
  }

  /**
   * Group results by question type (extra breakdown vs original evaluator)
   */
  groupByType(results) {
    const grouped = {};
    for (const result of results) {
      const t = result.type;
      if (!grouped[t]) grouped[t] = { count: 0, avgF1: 0, avgMRR: 0 };
      grouped[t].count++;
      grouped[t].avgF1 += result.f1Score;
      grouped[t].avgMRR += result.mrr;
    }
    for (const t in grouped) {
      const c = grouped[t].count;
      grouped[t].avgF1 = Math.round(grouped[t].avgF1 / c * 100) / 100;
      grouped[t].avgMRR = Math.round(grouped[t].avgMRR / c * 100) / 100;
    }
    return grouped;
  }

  /**
   * Content-specific test queries for 11-plus-syllabus.pdf.
   * Questions are grounded in the actual document text so recall failures
   * point to real retrieval gaps (chunk cap, embedding quality, etc.)
   */
  static createTestQueries() {
    const DOC = '11-plus-syllabus.pdf';

    return [
      // --- EASY: facts stated in the first ~3 chunks ---
      {
        question: "What subjects are included in the ISEB Common Pre-Test?",
        expectedSources: [DOC],
        difficulty: 'easy',
        type: 'factual'
      },
      {
        question: "How long does the Common Pre-Test take to complete?",
        expectedSources: [DOC],
        difficulty: 'easy',
        type: 'factual'
      },
      {
        question: "How long is the English comprehension paper?",
        expectedSources: [DOC],
        difficulty: 'easy',
        type: 'factual'
      },
      {
        question: "How long is the mathematics paper and is a calculator allowed?",
        expectedSources: [DOC],
        difficulty: 'easy',
        type: 'factual'
      },

      // --- MEDIUM: requires finding specific sections mid-document ---
      {
        question: "What are the three sections of the maths paper and what does each test?",
        expectedSources: [DOC],
        difficulty: 'medium',
        type: 'section_detail'
      },
      {
        question: "What writing forms are accepted in the composition paper?",
        expectedSources: [DOC],
        difficulty: 'medium',
        type: 'section_detail'
      },
      {
        question: "What National Curriculum level are candidates expected to reach in number?",
        expectedSources: [DOC],
        difficulty: 'medium',
        type: 'section_detail'
      },
      {
        question: "What types of questions appear in the English comprehension paper?",
        expectedSources: [DOC],
        difficulty: 'medium',
        type: 'section_detail'
      },

      // --- HARD: content in the latter half of the doc (chunks 6-10+) ---
      // These will fail if the maxChunksPerDoc cap cuts off the document too early.
      {
        question: "What topics are explicitly NOT tested in the maths exam?",
        expectedSources: [DOC],
        difficulty: 'hard',
        type: 'exclusion'
      },
      {
        question: "What shape and space topics are covered in the syllabus?",
        expectedSources: [DOC],
        difficulty: 'hard',
        type: 'late_content'
      },
      {
        question: "How should candidates check their calculation results?",
        expectedSources: [DOC],
        difficulty: 'hard',
        type: 'late_content'
      },
      {
        question: "What handling data topics will not be tested?",
        expectedSources: [DOC],
        difficulty: 'hard',
        type: 'exclusion'
      }
    ];
  }

  /**
   * Print a readable report to console
   */
  generateReport(evaluation, configName = 'ElevenPlus') {
    console.log(`\n📊 11+ Retrieval Evaluation Report - ${configName}`);
    console.log('='.repeat(55));

    const o = evaluation.overall;
    console.log(`📈 Overall (${o.totalQueries} queries):`);
    console.log(`   Precision:  ${o.avgPrecision}`);
    console.log(`   Recall:     ${o.avgRecall}`);
    console.log(`   F1:         ${o.avgF1}`);
    console.log(`   MRR:        ${o.avgMRR}`);
    console.log(`   NDCG:       ${o.avgNDCG}`);
    console.log(`   Avg time:   ${o.avgRetrievalTime}ms`);

    console.log(`\n📊 By Difficulty:`);
    for (const diff in evaluation.byDifficulty) {
      const m = evaluation.byDifficulty[diff];
      console.log(`   ${diff.toUpperCase()} (${m.count}): F1=${m.avgF1}  MRR=${m.avgMRR}`);
    }

    console.log(`\n📊 By Type:`);
    for (const t in evaluation.byType) {
      const m = evaluation.byType[t];
      console.log(`   ${t} (${m.count}): F1=${m.avgF1}  MRR=${m.avgMRR}`);
    }

    console.log(`\n📋 Per-Query Detail:`);
    for (const r of evaluation.detailed) {
      const pass = r.recall === 1 ? '✅' : '❌';
      console.log(`   ${pass} [${r.difficulty}] "${r.question}"`);
      console.log(`      F1=${r.f1Score}  MRR=${r.mrr}  chunks_retrieved=${r.retrievedCount}  relevant=${r.relevantCount}`);
    }

    return evaluation;
  }
}

module.exports = ElevenPlusEvaluator;
