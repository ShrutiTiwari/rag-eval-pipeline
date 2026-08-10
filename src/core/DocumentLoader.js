const fs = require('fs').promises;
const path = require('path');
const pdf = require('pdf-parse');
const { createChunkingStrategy } = require('../providers/ChunkingStrategy');

/**
 * DocumentLoader - Handles loading and processing of PDF documents
 * for the RAG chatbot system
 */
class DocumentLoader {
  /**
   * Load all PDF documents from a specified directory
   * @param {string} documentsPath - Path to directory containing PDFs
   * @returns {Promise<Array>} Array of document objects with content and metadata
   */
  static async loadDocuments(documentsPath = null) {
    try {
      // Default to public/pdf-sources if no path provided
      const docsPath = documentsPath || path.join(__dirname, '../../public/pdf-sources');

      console.log(`📁 Loading documents from: ${docsPath}`);

      // Check if directory exists
      try {
        await fs.access(docsPath);
      } catch (error) {
        console.warn(`⚠️  Directory not found: ${docsPath}`);
        return [];
      }

      // Read directory contents
      const files = await fs.readdir(docsPath);
      const pdfFiles = files.filter(file => path.extname(file).toLowerCase() === '.pdf');

      console.log(`📄 Found ${pdfFiles.length} PDF files`);

      if (pdfFiles.length === 0) {
        console.warn('⚠️  No PDF files found in directory');
        return [];
      }

      // Process each PDF file
      const documents = [];
      for (const filename of pdfFiles) {
        try {
          const filePath = path.join(docsPath, filename);
          const document = await this.loadSingleDocument(filePath);
          documents.push(document);
          console.log(`✅ Loaded: ${filename} (${document.content.length} chars)`);
        } catch (error) {
          console.error(`❌ Failed to load ${filename}:`, error.message);
          // Continue with other files even if one fails
        }
      }

      return documents;
    } catch (error) {
      console.error('❌ Error loading documents:', error.message);
      throw new Error(`Failed to load documents: ${error.message}`);
    }
  }

  /**
   * Load and process a single PDF document
   * @param {string} filePath - Path to the PDF file
   * @returns {Promise<Object>} Document object with content and metadata
   */
  static async loadSingleDocument(filePath) {
    try {
      // Read file buffer
      const fileBuffer = await fs.readFile(filePath);
      const fileStats = await fs.stat(filePath);

      // Extract text from PDF
      const pdfData = await pdf(fileBuffer);

      // Validate extracted content
      if (!pdfData.text || pdfData.text.trim().length === 0) {
        throw new Error('No text content extracted from PDF');
      }

      // Create document object
      const document = {
        content: this.cleanText(pdfData.text),
        metadata: {
          filename: path.basename(filePath),
          filepath: filePath,
          filesize: fileStats.size,
          pages: pdfData.numpages || 0,
          createdAt: fileStats.birthtime,
          modifiedAt: fileStats.mtime,
          extractedAt: new Date().toISOString()
        }
      };

      return document;
    } catch (error) {
      console.error(`❌ Error processing ${filePath}:`, error.message);
      throw new Error(`Failed to process PDF: ${error.message}`);
    }
  }

  /**
   * Clean and normalize extracted text
   * @param {string} text - Raw text from PDF
   * @returns {string} Cleaned text
   */
  static cleanText(text) {
    if (!text || typeof text !== 'string') {
      return '';
    }

    return text
      // Normalize line endings first
      .replace(/\r\n/g, '\n')
      .replace(/\r/g, '\n')
      // Remove multiple consecutive newlines
      .replace(/\n{3,}/g, '\n\n')
      // Remove excessive whitespace but preserve single newlines
      .replace(/[ \t]+/g, ' ')
      // Remove control characters except newlines and tabs
      .replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, '')
      // Trim whitespace
      .trim();
  }

  /**
   * Split document content into chunks using the configured chunking strategy.
   * @param {string} content - Document content
   * @param {number} chunkSize - Max chunk size (used by FixedSizeChunking)
   * @param {number} overlap - Overlap size (used by FixedSizeChunking)
   * @param {import('../providers/ChunkingStrategy').ChunkingStrategy} [strategy]
   *   Optional strategy instance. Defaults to createChunkingStrategy() which
   *   reads CHUNKING_STRATEGY env var ('fixed' or 'section').
   * @returns {Array<Object>} Array of chunk objects
   */
  static async chunkDocument(content, chunkSize = 1000, overlap = 100, strategy = null) {
    if (!content || typeof content !== 'string') {
      return [];
    }

    const chunkingStrategy = strategy || createChunkingStrategy(
      process.env.CHUNKING_STRATEGY || 'fixed',
      { chunkSize, overlap, maxSectionSize: chunkSize, }
    );

    return chunkingStrategy.chunk(content);
  }

  /**
   * Get document statistics
   * @param {Array} documents - Array of document objects
   * @returns {Object} Statistics about the documents
   */
  static getDocumentStats(documents) {
    if (!Array.isArray(documents) || documents.length === 0) {
      return {
        count: 0,
        totalSize: 0,
        totalPages: 0,
        totalCharacters: 0,
        avgCharsPerDoc: 0
      };
    }

    const stats = documents.reduce((acc, doc) => {
      acc.totalSize += doc.metadata?.filesize || 0;
      acc.totalPages += doc.metadata?.pages || 0;
      acc.totalCharacters += doc.content?.length || 0;
      return acc;
    }, {
      count: documents.length,
      totalSize: 0,
      totalPages: 0,
      totalCharacters: 0
    });

    stats.avgCharsPerDoc = Math.round(stats.totalCharacters / stats.count);

    return stats;
  }
}

module.exports = DocumentLoader;