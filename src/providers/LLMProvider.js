/**
 * LLMProvider - Base class and concrete implementations for LLM chat completions.
 * Swap the LLM by setting LLM_PROVIDER=claude (or 'openai') in your .env.
 *
 * NOTE: ClaudeLLMProvider requires the @anthropic-ai/sdk package.
 * Install it with: npm install @anthropic-ai/sdk
 */

// ---------------------------------------------------------------------------
// Base class
// ---------------------------------------------------------------------------
class LLMProvider {
  /**
   * Send a chat request and return the assistant's reply as a string.
   * @param {Array<{role: string, content: string}>} messages  OpenAI-style messages array
   * @param {Object} options
   * @param {number} [options.temperature=0.7]
   * @param {number} [options.maxTokens=1000]
   * @returns {Promise<string>}
   */
  async chat(messages, options = {}) {
    throw new Error('chat() must be implemented by a subclass');
  }
}

// ---------------------------------------------------------------------------
// OpenAI implementation
// ---------------------------------------------------------------------------
class OpenAILLMProvider extends LLMProvider {
  constructor(model = 'gpt-4o-mini') {
    super();
    this.model = model;
    this.apiKey = process.env.OPENAI_API_KEY;

    if (!this.apiKey) {
      throw new Error('OpenAILLMProvider requires OPENAI_API_KEY to be set in the environment');
    }

    const OpenAI = require('openai');
    this.client = new OpenAI({ apiKey: this.apiKey });
  }

  async chat(messages, options = {}) {
    const { temperature = 0.7, maxTokens = 1000 } = options;

    try {
      const response = await this.client.chat.completions.create({
        model: this.model,
        messages,
        temperature,
        max_tokens: maxTokens,
        presence_penalty: 0.1,
        frequency_penalty: 0.1
      });

      return response.choices[0].message.content;
    } catch (error) {
      throw new Error(`OpenAILLMProvider.chat() failed: ${error.message}`);
    }
  }
}

// ---------------------------------------------------------------------------
// Claude (Anthropic) implementation
// ---------------------------------------------------------------------------
class ClaudeLLMProvider extends LLMProvider {
  /**
   * @param {string} model  Anthropic model ID.
   *   Cheapest/fastest option: 'claude-haiku-4-5-20251001'
   */
  constructor(model = 'claude-haiku-4-5-20251001') {
    super();
    this.model = model;
    this.apiKey = process.env.ANTHROPIC_API_KEY;

    if (!this.apiKey) {
      throw new Error('ClaudeLLMProvider requires ANTHROPIC_API_KEY to be set in the environment');
    }

    // @anthropic-ai/sdk must be installed: npm install @anthropic-ai/sdk
    try {
      const Anthropic = require('@anthropic-ai/sdk');
      this.client = new Anthropic({ apiKey: this.apiKey });
    } catch (err) {
      throw new Error(
        'ClaudeLLMProvider requires @anthropic-ai/sdk. Install it with: npm install @anthropic-ai/sdk'
      );
    }
  }

  /**
   * Maps the OpenAI-style messages array to Anthropic format.
   * Anthropic separates the system prompt from the human/assistant turns.
   */
  _convertMessages(messages) {
    let systemPrompt = '';
    const anthropicMessages = [];

    for (const msg of messages) {
      if (msg.role === 'system') {
        // Anthropic takes system as a top-level param; concatenate if multiple
        systemPrompt += (systemPrompt ? '\n\n' : '') + msg.content;
      } else if (msg.role === 'user') {
        anthropicMessages.push({ role: 'user', content: msg.content });
      } else if (msg.role === 'assistant') {
        anthropicMessages.push({ role: 'assistant', content: msg.content });
      }
    }

    return { systemPrompt, anthropicMessages };
  }

  async chat(messages, options = {}) {
    const { temperature = 0.7, maxTokens = 1000 } = options;

    try {
      const { systemPrompt, anthropicMessages } = this._convertMessages(messages);

      const requestParams = {
        model: this.model,
        max_tokens: maxTokens,
        temperature,
        messages: anthropicMessages
      };

      if (systemPrompt) {
        requestParams.system = systemPrompt;
      }

      const response = await this.client.messages.create(requestParams);

      // Anthropic returns content as an array of blocks; extract the first text block
      const textBlock = response.content.find(block => block.type === 'text');
      if (!textBlock) {
        throw new Error('No text content returned by Anthropic API');
      }

      return textBlock.text;
    } catch (error) {
      throw new Error(`ClaudeLLMProvider.chat() failed: ${error.message}`);
    }
  }
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------
function createLLMProvider(type = process.env.LLM_PROVIDER || 'openai') {
  if (type === 'claude') return new ClaudeLLMProvider();
  return new OpenAILLMProvider();
}

module.exports = { createLLMProvider, OpenAILLMProvider, ClaudeLLMProvider };
