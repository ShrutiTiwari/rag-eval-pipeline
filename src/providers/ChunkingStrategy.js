/**
 * ChunkingStrategy - Base class and concrete implementations for document chunking.
 * Swap the strategy by setting CHUNKING_STRATEGY=section (or 'fixed') in your .env.
 *
 * All strategies produce the same chunk shape:
 *   { content, index, startChar, endChar, length, strategy, metadata? }
 */

// ---------------------------------------------------------------------------
// Base class
// ---------------------------------------------------------------------------
class ChunkingStrategy {
  /**
   * Split document content into chunks.
   * @param {string} content - Cleaned document text
   * @returns {Array<{content, index, startChar, endChar, length, strategy}>}
   */
  chunk(content) {
    throw new Error('chunk() must be implemented by a subclass');
  }

  /**
   * Build a chunk object with consistent shape.
   */
  _makeChunk(content, index, startChar, extraMeta = {}) {
    const trimmed = content.trim();
    return {
      content: trimmed,
      index,
      startChar,
      endChar: startChar + content.length,
      length: trimmed.length,
      strategy: this.constructor.name,
      ...extraMeta
    };
  }
}

// ---------------------------------------------------------------------------
// Strategy 1: FixedSizeChunking  (current default)
// ---------------------------------------------------------------------------
/**
 * Splits text into fixed-size overlapping windows.
 * Tries to break at sentence boundaries (. or \n) within the last 30% of each window
 * to avoid cutting mid-sentence.
 *
 * Good for: generic documents, uniform text density.
 * Weakness: ignores document structure — can blend multiple sections into one chunk.
 */
class FixedSizeChunking extends ChunkingStrategy {
  /**
   * @param {number} chunkSize  Max characters per chunk (default 1000)
   * @param {number} overlap    Overlap between consecutive chunks (default 100)
   */
  constructor(chunkSize = 1000, overlap = 100) {
    super();
    this.chunkSize = chunkSize;
    this.overlap = overlap;
  }

  chunk(content) {
    if (!content || typeof content !== 'string') return [];

    const chunks = [];
    let startIndex = 0;

    while (startIndex < content.length) {
      const endIndex = Math.min(startIndex + this.chunkSize, content.length);
      let chunkText = content.slice(startIndex, endIndex);

      // Try to break at sentence boundaries if not at end of document
      if (endIndex < content.length) {
        const lastPeriod  = chunkText.lastIndexOf('.');
        const lastNewline = chunkText.lastIndexOf('\n');
        const breakPoint  = Math.max(lastPeriod, lastNewline);

        if (breakPoint > chunkText.length * 0.7) {
          chunkText = content.slice(startIndex, startIndex + breakPoint + 1);
        }
      }

      chunks.push(this._makeChunk(chunkText, chunks.length, startIndex));

      // Advance with overlap
      startIndex = startIndex + chunkText.length - this.overlap;

      // Guard: always make forward progress
      if (startIndex <= chunks[chunks.length - 1].startChar) {
        startIndex = chunks[chunks.length - 1].endChar;
      }
    }

    console.log(`📝 FixedSizeChunking: ${chunks.length} chunks (size=${this.chunkSize}, overlap=${this.overlap})`);
    return chunks;
  }
}

// ---------------------------------------------------------------------------
// Strategy 2: SectionAwareChunking
// ---------------------------------------------------------------------------
/**
 * Splits text at recognised section headers, keeping each section's content
 * together in one or more chunks. Within sections that exceed maxSectionSize,
 * applies the same sentence-boundary splitting as FixedSizeChunking.
 *
 * Good for: structured documents with clear headings (syllabuses, reports, manuals).
 * Why this fixes the 11+ problem: "Mathematics" and "English" sections no longer
 * bleed into the same embedding — each section gets its own chunk(s).
 *
 * Recognised headers (case-insensitive, on their own line):
 *   - ALL CAPS words (e.g. NUMBER & THE NUMBER SYSTEM, CALCULATIONS)
 *   - Title Case lines followed by a blank line
 *   - Part A / Part B style labels
 *   - Lines ending with a colon that are short (< 60 chars)
 */
class SectionAwareChunking extends ChunkingStrategy {
  /**
   * @param {number} maxSectionSize  Max chars before a section is split further (default 1500)
   * @param {number} overlap         Overlap when splitting large sections (default 150)
   */
  constructor(maxSectionSize = 1500, overlap = 150) {
    super();
    this.maxSectionSize = maxSectionSize;
    this.overlap = overlap;

    // Patterns that indicate a new section boundary
    this.headerPatterns = [
      /^[A-Z][A-Z\s&,\/]{4,}$/,              // ALL CAPS line (min 5 chars): CALCULATIONS, NUMBER & THE NUMBER SYSTEM
      /^Part\s+[A-Z]:/i,                       // Part A: / Part B:
      /^(Mathematics|English|Science|History|Geography|Introduction|Summary|Conclusion)\s*$/i,
      /^Section\s+[A-Z0-9]/i,                  // Section A, Section 1
      /^#{1,3}\s+/,                            // Markdown headings (future-proofing)
    ];
  }

  _isHeader(line) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.length < 3) return false;
    return this.headerPatterns.some(p => p.test(trimmed));
  }

  /**
   * Split a section's text into sub-chunks if it exceeds maxSectionSize.
   * Uses sentence-boundary splitting with overlap — same logic as FixedSizeChunking.
   */
  _splitLargeSection(text, sectionLabel) {
    const subChunks = [];
    let start = 0;

    while (start < text.length) {
      const end = Math.min(start + this.maxSectionSize, text.length);
      let slice = text.slice(start, end);

      if (end < text.length) {
        const lastPeriod  = slice.lastIndexOf('.');
        const lastNewline = slice.lastIndexOf('\n');
        const breakPoint  = Math.max(lastPeriod, lastNewline);
        if (breakPoint > slice.length * 0.7) {
          slice = text.slice(start, start + breakPoint + 1);
        }
      }

      subChunks.push({ text: slice.trim(), startOffset: start });
      start = start + slice.length - this.overlap;
      if (start <= subChunks[subChunks.length - 1].startOffset) {
        start = subChunks[subChunks.length - 1].startOffset + slice.length;
      }
    }

    return subChunks;
  }

  chunk(content) {
    if (!content || typeof content !== 'string') return [];

    const lines = content.split('\n');
    const sections = [];   // { label, text, startChar }
    let currentLabel = 'Introduction';
    let currentLines = [];
    let charPos = 0;
    let sectionStart = 0;

    for (const line of lines) {
      if (this._isHeader(line)) {
        // Save current section
        if (currentLines.length > 0) {
          const text = currentLines.join('\n');
          if (text.trim().length > 0) {
            sections.push({ label: currentLabel, text, startChar: sectionStart });
          }
        }
        // Start new section
        currentLabel = line.trim();
        currentLines = [line];
        sectionStart = charPos;
      } else {
        currentLines.push(line);
      }
      charPos += line.length + 1; // +1 for the \n
    }

    // Push last section
    if (currentLines.length > 0) {
      const text = currentLines.join('\n');
      if (text.trim().length > 0) {
        sections.push({ label: currentLabel, text, startChar: sectionStart });
      }
    }

    // Convert sections to chunks
    const chunks = [];

    for (const section of sections) {
      if (section.text.length <= this.maxSectionSize) {
        // Section fits in one chunk
        chunks.push(this._makeChunk(section.text, chunks.length, section.startChar, {
          section: section.label
        }));
      } else {
        // Section is too large — split it further
        const subChunks = this._splitLargeSection(section.text, section.label);
        for (const sub of subChunks) {
          chunks.push(this._makeChunk(sub.text, chunks.length, section.startChar + sub.startOffset, {
            section: section.label
          }));
        }
      }
    }

    console.log(`📝 SectionAwareChunking: ${chunks.length} chunks from ${sections.length} sections (maxSize=${this.maxSectionSize})`);
    return chunks;
  }
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------
function createChunkingStrategy(type = process.env.CHUNKING_STRATEGY || 'fixed', options = {}) {
  if (type === 'section') {
    return new SectionAwareChunking(
      options.maxSectionSize || 1500,
      options.overlap || 150
    );
  }
  return new FixedSizeChunking(
    options.chunkSize || 1000,
    options.overlap || 100
  );
}

module.exports = { createChunkingStrategy, FixedSizeChunking, SectionAwareChunking };
