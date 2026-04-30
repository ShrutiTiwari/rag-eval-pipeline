/**
 * TruLens Client - Interface to TruLens Python evaluation service
 */
class TruLensClient {
  constructor(baseUrl = 'http://localhost:5001') {
    this.baseUrl = baseUrl;
  }

  /**
   * Check if TruLens service is available
   */
  async isAvailable() {
    try {
      const response = await fetch(`${this.baseUrl}/health`);
      const data = await response.json();
      return data.status === 'healthy';
    } catch (error) {
      console.warn('⚠️ TruLens service not available:', error.message);
      return false;
    }
  }

  /**
   * Evaluate a single RAG response using TruLens
   * @param {string} query - User question
   * @param {Array} retrievedChunks - Retrieved document chunks
   * @param {string} generatedResponse - RAG system response
   * @param {Object} metadata - Additional metadata
   * @returns {Promise<Object>} TruLens evaluation results
   */
  async evaluateSingle(query, retrievedChunks, generatedResponse, metadata = {}) {
    try {
      const payload = {
        query,
        retrieved_contexts: retrievedChunks.map(chunk =>
          typeof chunk === 'string' ? chunk : chunk.content
        ),
        generated_response: generatedResponse,
        metadata: {
          ...metadata,
          timestamp: new Date().toISOString()
        }
      };

      const response = await fetch(`${this.baseUrl}/evaluate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        throw new Error(`TruLens API error: ${response.status}`);
      }

      const data = await response.json();

      if (!data.success) {
        throw new Error(data.error || 'TruLens evaluation failed');
      }

      return data.result;

    } catch (error) {
      console.error('❌ TruLens evaluation error:', error.message);
      throw new Error(`TruLens evaluation failed: ${error.message}`);
    }
  }

  /**
   * Evaluate multiple RAG responses in batch
   * @param {Array} evaluations - Array of evaluation requests
   * @returns {Promise<Object>} Batch evaluation results
   */
  async evaluateBatch(evaluations) {
    try {
      const payload = {
        evaluations: evaluations.map(item => ({
          query: item.query,
          retrieved_contexts: item.retrievedChunks.map(chunk =>
            typeof chunk === 'string' ? chunk : chunk.content
          ),
          generated_response: item.generatedResponse,
          metadata: item.metadata || {}
        }))
      };

      const response = await fetch(`${this.baseUrl}/evaluate/batch`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        throw new Error(`TruLens API error: ${response.status}`);
      }

      const data = await response.json();

      if (!data.success) {
        throw new Error(data.error || 'TruLens batch evaluation failed');
      }

      return data.result;

    } catch (error) {
      console.error('❌ TruLens batch evaluation error:', error.message);
      throw new Error(`TruLens batch evaluation failed: ${error.message}`);
    }
  }

  /**
   * Get available feedback functions
   * @returns {Promise<Object>} Available feedback functions
   */
  async getFeedbackFunctions() {
    try {
      const response = await fetch(`${this.baseUrl}/feedback-functions`);
      const data = await response.json();
      return data.feedback_functions;
    } catch (error) {
      console.error('❌ Error fetching feedback functions:', error.message);
      return {};
    }
  }

  /**
   * Format TruLens results for display
   * @param {Object} truLensResult - Raw TruLens evaluation result
   * @returns {Object} Formatted results
   */
  formatResults(truLensResult) {
    const formatted = {
      scores: {},
      reasons: {},
      overallScore: truLensResult.overall_score || 0,
      metadata: truLensResult.metadata || {}
    };

    // Extract scores and reasons
    const metrics = [
      'groundedness',
      'answer_relevance',
      'context_relevance',
      'coherence',
      'retrieval_precision'
    ];

    for (const metric of metrics) {
      if (truLensResult[metric]) {
        formatted.scores[metric] = truLensResult[metric].score || 0;
        formatted.reasons[metric] = truLensResult[metric].reason || null;
      }
    }

    return formatted;
  }

  /**
   * Get score interpretation
   * @param {number} score - Score between 0 and 1
   * @returns {Object} Score interpretation
   */
  getScoreInterpretation(score) {
    if (score >= 0.8) {
      return { level: 'excellent', color: '#10b981', description: 'Excellent performance' };
    } else if (score >= 0.6) {
      return { level: 'good', color: '#3b82f6', description: 'Good performance' };
    } else if (score >= 0.4) {
      return { level: 'fair', color: '#f59e0b', description: 'Fair performance' };
    } else if (score >= 0.2) {
      return { level: 'poor', color: '#ef4444', description: 'Poor performance' };
    } else {
      return { level: 'very-poor', color: '#991b1b', description: 'Very poor performance' };
    }
  }
}

module.exports = TruLensClient;