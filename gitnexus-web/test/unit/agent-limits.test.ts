import { describe, expect, it, vi } from 'vitest';
import { BaseChatModel } from '@langchain/core/language_models/chat_models';
import { createReactAgent } from '@langchain/langgraph/prebuilt';
import { tool } from '@langchain/core/tools';
import { AIMessage } from '@langchain/core/messages';
import { ChatGoogleGenerativeAI } from '@langchain/google-genai';
import { z } from 'zod';
import {
  streamAgentResponse,
  toRecursionLimit,
  extractMessageUsage,
} from '../../src/core/llm/agent';
import type { AgentStreamChunk } from '../../src/core/llm/types';

type MockAgent = Parameters<typeof streamAgentResponse>[0];

function makeAiChunk(opts: {
  id?: string;
  content?: string;
  usage_metadata?: { input_tokens?: number; output_tokens?: number; total_tokens?: number };
  response_metadata?: Record<string, unknown>;
  tool_calls?: unknown[];
}) {
  return {
    _getType: () => 'ai',
    id: opts.id,
    content: opts.content ?? '',
    tool_calls: opts.tool_calls ?? [],
    usage_metadata: opts.usage_metadata,
    response_metadata: opts.response_metadata,
  };
}

describe('toRecursionLimit', () => {
  it('defaults to 17 for 8 iterations when omitted', () => {
    expect(toRecursionLimit()).toBe(17);
    expect(toRecursionLimit(undefined)).toBe(17);
    expect(toRecursionLimit(Number.NaN)).toBe(17);
  });

  it('maps iterations to 2 * iterations + 1 within [1, 50]', () => {
    expect(toRecursionLimit(1)).toBe(3);
    expect(toRecursionLimit(8)).toBe(17);
    expect(toRecursionLimit(20)).toBe(41);
    expect(toRecursionLimit(50)).toBe(101);
  });

  it('defaults non-positive values and clamps upper out-of-range values', () => {
    expect(toRecursionLimit(0)).toBe(17);
    expect(toRecursionLimit(-5)).toBe(17);
    expect(toRecursionLimit(200)).toBe(101);
  });
});

describe('extractMessageUsage', () => {
  it('extracts LangChain usage_metadata', () => {
    expect(
      extractMessageUsage({
        usage_metadata: { input_tokens: 120, output_tokens: 45, total_tokens: 165 },
      }),
    ).toEqual({
      inputTokens: 120,
      outputTokens: 45,
      totalTokens: 165,
    });
  });

  it('extracts OpenAI-style response_metadata.tokenUsage', () => {
    expect(
      extractMessageUsage({
        response_metadata: {
          tokenUsage: { promptTokens: 80, completionTokens: 20 },
        },
      }),
    ).toEqual({
      inputTokens: 80,
      outputTokens: 20,
      totalTokens: 100,
    });
  });

  it('returns null when usage is absent or zero', () => {
    expect(extractMessageUsage(null)).toBeNull();
    expect(extractMessageUsage({})).toBeNull();
    expect(
      extractMessageUsage({
        usage_metadata: { input_tokens: 0, output_tokens: 0, total_tokens: 0 },
      }),
    ).toBeNull();
  });
});

describe('streamAgentResponse limits and token tracking', () => {
  it('passes computed recursionLimit to agent.stream', async () => {
    let capturedConfig: Record<string, unknown> | undefined;
    const fakeAgent = {
      stream: vi.fn(async (_input: unknown, config: Record<string, unknown>) => {
        capturedConfig = config;
        return (async function* () {
          yield ['messages', [makeAiChunk({ id: 'ai-1', content: 'hi' }), {}]];
        })();
      }),
    } as unknown as MockAgent;

    const chunks: AgentStreamChunk[] = [];
    for await (const chunk of streamAgentResponse(fakeAgent, [{ role: 'user', content: 'hello' }], {
      maxIterations: 12,
    })) {
      chunks.push(chunk);
    }

    expect(capturedConfig?.recursionLimit).toBe(25);
    expect(chunks.at(-1)).toEqual({ type: 'done', historyMessages: undefined });
  });

  it('deduplicates usage between messages and values snapshots and sums across turns', async () => {
    const fakeAgent = {
      stream: vi.fn(async () => {
        return (async function* () {
          yield [
            'messages',
            [
              makeAiChunk({
                id: 'ai-1',
                content: 'Thinking...',
                usage_metadata: { input_tokens: 100, output_tokens: 5, total_tokens: 105 },
              }),
              {},
            ],
          ];
          yield [
            'messages',
            [
              makeAiChunk({
                id: 'ai-1',
                content: '',
                usage_metadata: { input_tokens: 100, output_tokens: 25, total_tokens: 125 },
              }),
              {},
            ],
          ];
          yield [
            'values',
            {
              messages: [
                { _getType: () => 'human', content: 'hello' },
                makeAiChunk({
                  id: 'ai-1',
                  content: 'Thinking...',
                  usage_metadata: { input_tokens: 100, output_tokens: 25, total_tokens: 125 },
                }),
              ],
            },
          ];
          yield [
            'messages',
            [
              makeAiChunk({
                id: 'ai-2',
                content: 'Answer',
                usage_metadata: { input_tokens: 40, output_tokens: 20, total_tokens: 60 },
              }),
              {},
            ],
          ];
        })();
      }),
    } as unknown as MockAgent;

    const chunks: AgentStreamChunk[] = [];
    for await (const chunk of streamAgentResponse(fakeAgent, [
      { role: 'user', content: 'hello' },
    ])) {
      chunks.push(chunk);
    }

    const doneChunk = chunks.find((c) => c.type === 'done');
    expect(doneChunk).toEqual({
      type: 'done',
      historyMessages: undefined,
      usage: {
        inputTokens: 140,
        outputTokens: 45,
        totalTokens: 185,
      },
    });
  });

  it('gracefully halts and yields budget_exceeded when cumulative tokens exceed tokenBudget', async () => {
    let capturedSignal: AbortSignal | undefined;
    const fakeAgent = {
      stream: vi.fn(async (_input: unknown, config: { signal?: AbortSignal }) => {
        capturedSignal = config.signal;
        return (async function* () {
          yield [
            'messages',
            [
              makeAiChunk({
                id: 'ai-1',
                content: 'step 1',
                usage_metadata: { input_tokens: 80, output_tokens: 30, total_tokens: 110 },
              }),
              {},
            ],
          ];
          yield [
            'messages',
            [
              makeAiChunk({
                id: 'ai-2',
                content: 'step 2',
                usage_metadata: { input_tokens: 100, output_tokens: 50, total_tokens: 150 },
              }),
              {},
            ],
          ];
          yield ['messages', [makeAiChunk({ id: 'ai-3', content: 'should not be yielded' }), {}]];
        })();
      }),
    } as unknown as MockAgent;

    const userController = new AbortController();
    const chunks: AgentStreamChunk[] = [];
    for await (const chunk of streamAgentResponse(
      fakeAgent,
      [{ role: 'user', content: 'explain architecture' }],
      { signal: userController.signal, tokenBudget: 200 },
    )) {
      chunks.push(chunk);
    }

    expect(capturedSignal?.aborted).toBe(true);
    expect(userController.signal.aborted).toBe(false);
    expect(chunks.some((c) => c.type === 'cancelled')).toBe(false);
    expect(chunks.at(-1)).toEqual({
      type: 'budget_exceeded',
      reason: 'token_budget',
      limit: 200,
      usage: {
        inputTokens: 180,
        outputTokens: 80,
        totalTokens: 260,
      },
    });
  });

  it('yields cancelled instead of budget_exceeded when user aborts during a budgeted run', async () => {
    const userController = new AbortController();
    const fakeAgent = {
      stream: vi.fn(async (_input: unknown, config: { signal?: AbortSignal }) => {
        return (async function* () {
          yield [
            'messages',
            [
              makeAiChunk({
                id: 'ai-1',
                content: 'partial',
                usage_metadata: { input_tokens: 50, output_tokens: 10, total_tokens: 60 },
              }),
              {},
            ],
          ];
          userController.abort();
          if (config.signal?.aborted) {
            throw new DOMException('Aborted', 'AbortError');
          }
        })();
      }),
    } as unknown as MockAgent;

    const chunks: AgentStreamChunk[] = [];
    for await (const chunk of streamAgentResponse(fakeAgent, [{ role: 'user', content: 'hello' }], {
      signal: userController.signal,
      tokenBudget: 500,
    })) {
      chunks.push(chunk);
    }

    expect(chunks.at(-1)).toEqual({ type: 'cancelled' });
  });

  it('yields budget_exceeded when LangGraph throws recursion limit error', async () => {
    const fakeAgent = {
      stream: vi.fn(async () => {
        return (async function* () {
          yield [
            'messages',
            [
              makeAiChunk({
                id: 'ai-1',
                content: 'searching',
                usage_metadata: { input_tokens: 90, output_tokens: 30, total_tokens: 120 },
              }),
              {},
            ],
          ];
          const err = new Error('Recursion limit of 17 reached without hitting a stop condition');
          Object.assign(err, { lc_error_code: 'GRAPH_RECURSION_LIMIT' });
          throw err;
        })();
      }),
    } as unknown as MockAgent;

    const chunks: AgentStreamChunk[] = [];
    for await (const chunk of streamAgentResponse(
      fakeAgent,
      [{ role: 'user', content: 'deep trace' }],
      { maxIterations: 8 },
    )) {
      chunks.push(chunk);
    }

    expect(chunks.at(-1)).toEqual({
      type: 'budget_exceeded',
      reason: 'max_iterations',
      limit: 8,
      usage: {
        inputTokens: 90,
        outputTokens: 30,
        totalTokens: 120,
      },
    });
  });

  it('completes without stopping and flags usageUnavailable when a budgeted stream emits text with no usage metadata', async () => {
    const fakeAgent = {
      stream: vi.fn(async () => {
        return (async function* () {
          yield [
            'messages',
            [
              makeAiChunk({
                id: 'ai-1',
                content: 'Response without provider token metadata',
              }),
              {},
            ],
          ];
          yield [
            'values',
            {
              messages: [
                { _getType: () => 'human', content: 'explain architecture' },
                makeAiChunk({
                  id: 'ai-1',
                  content: 'Response without provider token metadata',
                }),
              ],
            },
          ];
        })();
      }),
    } as unknown as MockAgent;

    const userController = new AbortController();
    const chunks: AgentStreamChunk[] = [];
    for await (const chunk of streamAgentResponse(
      fakeAgent,
      [{ role: 'user', content: 'explain architecture' }],
      { signal: userController.signal, tokenBudget: 50 },
    )) {
      chunks.push(chunk);
    }

    expect(userController.signal.aborted).toBe(false);
    expect(chunks).toEqual([
      { type: 'reasoning', reasoning: 'Response without provider token metadata' },
      { type: 'done', historyMessages: undefined, usageUnavailable: true },
    ]);
  });

  it('accumulates incremental deltas for Gemini streams and halts when budget is reached', async () => {
    let capturedSignal: AbortSignal | undefined;
    const fakeAgent = {
      stream: vi.fn(async (_input: unknown, config: { signal?: AbortSignal }) => {
        capturedSignal = config.signal;
        return (async function* () {
          // Chunk 1: Initial prompt tokens + first candidate tokens
          yield [
            'messages',
            [
              makeAiChunk({
                id: 'gemini-turn-1',
                content: 'Step 1',
                response_metadata: { model_provider: 'google-genai' },
                usage_metadata: { input_tokens: 100, output_tokens: 10, total_tokens: 110 },
              }),
              {},
            ],
          ];
          // Chunk 2: Incremental delta (5 new candidate tokens) -> total should be 115
          yield [
            'messages',
            [
              makeAiChunk({
                id: 'gemini-turn-1',
                content: 'Step 2',
                response_metadata: { model_provider: 'google-genai' },
                usage_metadata: { input_tokens: 0, output_tokens: 5, total_tokens: 5 },
              }),
              {},
            ],
          ];
          // Chunk 3: Incremental delta (10 new candidate tokens) -> total becomes 125, exceeding 115 budget
          yield [
            'messages',
            [
              makeAiChunk({
                id: 'gemini-turn-1',
                content: 'Step 3',
                response_metadata: { model_provider: 'google-genai' },
                usage_metadata: { input_tokens: 0, output_tokens: 10, total_tokens: 10 },
              }),
              {},
            ],
          ];
          // Chunk 4: Should not be processed because budget was exceeded on chunk 3
          yield [
            'messages',
            [
              makeAiChunk({
                id: 'gemini-turn-1',
                content: 'Step 4',
                response_metadata: { model_provider: 'google-genai' },
                usage_metadata: { input_tokens: 0, output_tokens: 10, total_tokens: 10 },
              }),
              {},
            ],
          ];
        })();
      }),
    } as unknown as MockAgent;

    const chunks: AgentStreamChunk[] = [];
    for await (const chunk of streamAgentResponse(
      fakeAgent,
      [{ role: 'user', content: 'run query' }],
      { tokenBudget: 115 },
    )) {
      chunks.push(chunk);
    }

    expect(capturedSignal?.aborted).toBe(true);
    expect(chunks.some((c) => c.type === 'reasoning' && c.reasoning.includes('Step 4'))).toBe(
      false,
    );
    expect(chunks.at(-1)).toEqual({
      type: 'budget_exceeded',
      reason: 'token_budget',
      limit: 115,
      usage: {
        inputTokens: 100,
        outputTokens: 25,
        totalTokens: 125,
      },
    });
  });

  it('accumulates Gemini deltas and reconciles once with final values snapshot without double counting', async () => {
    const fakeAgent = {
      stream: vi.fn(async () => {
        return (async function* () {
          yield [
            'messages',
            [
              makeAiChunk({
                id: 'gemini-turn-1',
                content: 'Thinking',
                response_metadata: { model_provider: 'google-genai' },
                usage_metadata: { input_tokens: 100, output_tokens: 10, total_tokens: 110 },
              }),
              {},
            ],
          ];
          yield [
            'messages',
            [
              makeAiChunk({
                id: 'gemini-turn-1',
                content: ' more',
                response_metadata: { model_provider: 'google-genai' },
                usage_metadata: { input_tokens: 0, output_tokens: 15, total_tokens: 15 },
              }),
              {},
            ],
          ];
          // Final values mode snapshot with aggregate usage (100 + 25 = 125)
          yield [
            'values',
            {
              messages: [
                { _getType: () => 'human', content: 'run query' },
                makeAiChunk({
                  id: 'gemini-turn-1',
                  content: 'Thinking more',
                  response_metadata: { model_provider: 'google-genai' },
                  usage_metadata: { input_tokens: 100, output_tokens: 25, total_tokens: 125 },
                }),
              ],
            },
          ];
        })();
      }),
    } as unknown as MockAgent;

    const chunks: AgentStreamChunk[] = [];
    for await (const chunk of streamAgentResponse(
      fakeAgent,
      [{ role: 'user', content: 'run query' }],
      { tokenBudget: 200 },
    )) {
      chunks.push(chunk);
    }

    const doneChunk = chunks.find((c) => c.type === 'done');
    expect(doneChunk).toEqual({
      type: 'done',
      historyMessages: undefined,
      usage: {
        inputTokens: 100,
        outputTokens: 25,
        totalTokens: 125,
      },
    });
  });

  it('halts at exact equality on tool-to-model transition when remaining budget is zero', async () => {
    let capturedSignal: AbortSignal | undefined;
    const fakeAgent = {
      stream: vi.fn(async (_input: unknown, config: { signal?: AbortSignal }) => {
        capturedSignal = config.signal;
        return (async function* () {
          // Model turn 1: emits a tool call and exhausts budget exactly (100 / 100 tokens)
          yield [
            'messages',
            [
              makeAiChunk({
                id: 'ai-turn-1',
                content: '',
                tool_calls: [{ id: 'call-1', name: 'search_code', args: { q: 'auth' } }],
                usage_metadata: { input_tokens: 70, output_tokens: 30, total_tokens: 100 },
              }),
              {},
            ],
          ];
          // Tool executes and emits result
          yield [
            'messages',
            [
              {
                _getType: () => 'tool',
                tool_call_id: 'call-1',
                name: 'search_code',
                content: 'result',
              },
              {},
            ],
          ];
          // Subsequent model turn that should NOT execute
          yield [
            'messages',
            [
              makeAiChunk({
                id: 'ai-turn-2',
                content: 'Continuing with remaining budget 0...',
                usage_metadata: { input_tokens: 20, output_tokens: 10, total_tokens: 30 },
              }),
              {},
            ],
          ];
        })();
      }),
    } as unknown as MockAgent;

    const chunks: AgentStreamChunk[] = [];
    for await (const chunk of streamAgentResponse(
      fakeAgent,
      [{ role: 'user', content: 'search and summarize' }],
      { tokenBudget: 100 },
    )) {
      chunks.push(chunk);
    }

    expect(capturedSignal?.aborted).toBe(true);
    // Verified that tool call and tool result are processed before transition halts
    expect(chunks.some((c) => c.type === 'tool_call')).toBe(true);
    expect(chunks.some((c) => c.type === 'tool_result')).toBe(true);
    // Verified that it yields budget_exceeded at exact equality and does not process ai-turn-2
    expect(chunks.some((c) => c.type === 'reasoning' && c.reasoning.includes('Continuing'))).toBe(
      false,
    );
    expect(chunks.at(-1)).toEqual({
      type: 'budget_exceeded',
      reason: 'token_budget',
      limit: 100,
      usage: {
        inputTokens: 70,
        outputTokens: 30,
        totalTokens: 100,
      },
    });
  });

  it('does not discard final AI chunk content when usage lands exactly on the budget limit', async () => {
    const fakeAgent = {
      stream: vi.fn(async () => {
        return (async function* () {
          yield [
            'messages',
            [
              makeAiChunk({
                id: 'ai-turn-final',
                content: '42 is the answer.',
                usage_metadata: { input_tokens: 70, output_tokens: 30, total_tokens: 100 },
              }),
              {},
            ],
          ];
        })();
      }),
    } as unknown as MockAgent;

    const chunks: AgentStreamChunk[] = [];
    for await (const chunk of streamAgentResponse(
      fakeAgent,
      [{ role: 'user', content: 'what is the answer?' }],
      { tokenBudget: 100 },
    )) {
      chunks.push(chunk);
    }

    // Verify chunk content was yielded rather than discarded
    expect(
      chunks.some(
        (c) =>
          c.type === 'content' ||
          (c.type === 'reasoning' && c.reasoning.includes('42 is the answer.')),
      ),
    ).toBe(true);
    // Completes normally with done rather than budget_exceeded because budget was met, not exceeded
    const doneChunk = chunks.find((c) => c.type === 'done');
    expect(doneChunk).toEqual({
      type: 'done',
      historyMessages: undefined,
      usage: {
        inputTokens: 70,
        outputTokens: 30,
        totalTokens: 100,
      },
    });
  });

  it('pins tool-to-model transition with locked LangGraph 1.7.4 ensuring model is never invoked after budget exhaustion', async () => {
    class CountingChatModel extends BaseChatModel {
      _llmType() {
        return 'counting-model';
      }
      modelCallCount = 0;
      constructor() {
        super({});
      }
      async _generate(_messages: any[], _options?: any) {
        this.modelCallCount++;
        if (this.modelCallCount === 1) {
          return {
            generations: [
              {
                text: '',
                message: new AIMessage({
                  content: '',
                  tool_calls: [{ id: 'call_1', name: 'search_tool', args: { query: 'test' } }],
                  usage_metadata: { input_tokens: 60, output_tokens: 40, total_tokens: 100 },
                }),
              },
            ],
          };
        }
        return {
          generations: [
            {
              text: 'Unwanted second turn',
              message: new AIMessage({ content: 'Unwanted second turn' }),
            },
          ],
        };
      }
      bindTools() {
        return this;
      }
    }

    const searchTool = tool(async () => 'search results', {
      name: 'search_tool',
      description: 'Search tool',
      schema: z.object({ query: z.string() }),
    });

    const model = new CountingChatModel();
    const agent = createReactAgent({ llm: model, tools: [searchTool] });

    const chunks: AgentStreamChunk[] = [];
    for await (const chunk of streamAgentResponse(
      agent,
      [{ role: 'user', content: 'search something' }],
      { tokenBudget: 100 },
    )) {
      chunks.push(chunk);
      if (chunk.type === 'tool_result') {
        // Consumer handles tool result asynchronously while at exact budget limit
        await new Promise((resolve) => setTimeout(resolve, 50));
      }
    }

    // Model invocation counter must strictly be 1 — never invoked a second time
    expect(model.modelCallCount).toBe(1);
    expect(chunks.some((c) => c.type === 'tool_call')).toBe(true);
    expect(chunks.some((c) => c.type === 'tool_result')).toBe(true);
    expect(chunks.some((c) => c.type === 'content' && c.content.includes('Unwanted'))).toBe(false);
    expect(chunks.at(-1)).toEqual({
      type: 'budget_exceeded',
      reason: 'token_budget',
      limit: 100,
      usage: {
        inputTokens: 60,
        outputTokens: 40,
        totalTokens: 100,
      },
    });
  });

  it('ignores carried-over usage metadata on metadata-free chunks with locked Google GenAI adapter', async () => {
    const model = new ChatGoogleGenerativeAI({
      apiKey: 'test-key',
      model: 'gemini-1.5-flash',
    });

    (model as any).client = {
      generateContentStream: async () => ({
        stream: (async function* () {
          // Chunk 1: provider reports usage metadata
          yield {
            candidates: [{ content: { role: 'model', parts: [{ text: 'Hello' }] } }],
            usageMetadata: { promptTokenCount: 10, candidatesTokenCount: 5, totalTokenCount: 15 },
          };
          // Chunk 2: metadata-free streaming text chunk (adapter will carry over chunk 1 usageMetadata)
          yield {
            candidates: [{ content: { role: 'model', parts: [{ text: ' world' }] } }],
          };
          // Chunk 3: another metadata-free chunk
          yield {
            candidates: [{ content: { role: 'model', parts: [{ text: '!' }] } }],
          };
        })(),
      }),
    };

    const stream = model._streamResponseChunks([], {});
    const fakeAgent = {
      stream: vi.fn(async () =>
        (async function* () {
          for await (const genChunk of stream) {
            yield ['messages', [genChunk.message, {}]];
          }
        })(),
      ),
    } as unknown as MockAgent;

    const chunks: AgentStreamChunk[] = [];
    for await (const chunk of streamAgentResponse(
      fakeAgent,
      [{ role: 'user', content: 'say hello' }],
      { tokenBudget: 25 },
    )) {
      chunks.push(chunk);
    }

    // Carried-over metadata was ignored; does not double-count to 45 (which would halt budget at 25)
    expect(chunks.some((c) => c.type === 'budget_exceeded')).toBe(false);
    const doneChunk = chunks.find((c) => c.type === 'done');
    expect(doneChunk).toEqual({
      type: 'done',
      historyMessages: undefined,
      usage: {
        inputTokens: 10,
        outputTokens: 5,
        totalTokens: 15,
      },
    });
  });

  it('flags usageUnavailable while preserving partial usage when one model step reports usage and another omits it', async () => {
    const fakeAgent = {
      stream: vi.fn(async () => {
        return (async function* () {
          // Step 1: Tool call with usage reported (50 tokens)
          yield [
            'messages',
            [
              makeAiChunk({
                id: 'ai-step-1',
                content: '',
                tool_calls: [{ id: 'call-1', name: 'search_code', args: { q: 'auth' } }],
                usage_metadata: { input_tokens: 30, output_tokens: 20, total_tokens: 50 },
              }),
              {},
            ],
          ];
          // Tool executes and emits result
          yield [
            'messages',
            [
              {
                _getType: () => 'tool',
                tool_call_id: 'call-1',
                name: 'search_code',
                content: 'result',
              },
              {},
            ],
          ];
          // Step 2: Final answer with usage omitted (provider didn't return usage)
          yield [
            'messages',
            [
              makeAiChunk({
                id: 'ai-step-2',
                content: 'Here is the summary of auth.',
                usage_metadata: undefined,
              }),
              {},
            ],
          ];
        })();
      }),
    } as unknown as MockAgent;

    const chunks: AgentStreamChunk[] = [];
    for await (const chunk of streamAgentResponse(
      fakeAgent,
      [{ role: 'user', content: 'search and summarize' }],
      { tokenBudget: 200 },
    )) {
      chunks.push(chunk);
    }

    const doneChunk = chunks.find((c) => c.type === 'done');
    expect(doneChunk).toEqual({
      type: 'done',
      historyMessages: undefined,
      usage: {
        inputTokens: 30,
        outputTokens: 20,
        totalTokens: 50,
      },
      usageUnavailable: true,
    });
  });
});
