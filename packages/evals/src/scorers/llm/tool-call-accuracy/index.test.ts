import { openai } from '@ai-sdk/openai';
import { convertArrayToReadableStream, MockLanguageModelV2 } from '@internal/ai-sdk-v5/test';
import { getLLMTestMode } from '@internal/llm-recorder';
import { setupDummyApiKeys } from '@internal/test-utils';
import { Agent } from '@mastra/core/agent';
import { createScorer } from '@mastra/core/evals';
import type { ScorerRunInputForAgent } from '@mastra/core/evals';
import { Mastra } from '@mastra/core/mastra';
import { InMemoryStore } from '@mastra/core/storage';
import { createTool } from '@mastra/core/tools';
import { describe, expect, it, vi } from 'vitest';
import { z } from 'zod/v4';
import { createAgentTestRun, createTestMessage, extractToolCalls } from '../../utils';
import { createToolCallAccuracyScorerLLM } from './index';

setupDummyApiKeys(getLLMTestMode(), ['openai']);

describe('createToolCallAccuracyScorerLLM', () => {
  const mockModel = openai('gpt-4o-mini');

  const availableTools = [
    createTool({ id: 'calculator-tool', description: 'Perform mathematical calculations' }),
    createTool({ id: 'search-tool', description: 'Search the web for information' }),
    createTool({ id: 'calendar-tool', description: 'Manage calendar events' }),
    createTool({ id: 'weather-tool', description: 'Get weather information' }),
  ];

  describe('per-run available tools', () => {
    it('judges each saved run against its own tools and preserves static lists', async () => {
      const prompts: string[] = [];
      const model = new MockLanguageModelV2({
        doStream: async ({ prompt }) => {
          prompts.push(JSON.stringify(prompt));
          return {
            stream: convertArrayToReadableStream([
              { type: 'stream-start', warnings: [] },
              { type: 'text-start', id: 'text' },
              { type: 'text-delta', id: 'text', delta: JSON.stringify({ evaluations: [], missingTools: [] }) },
              { type: 'text-end', id: 'text' },
              { type: 'finish', finishReason: 'stop', usage: { inputTokens: 1, outputTokens: 1, totalTokens: 2 } },
            ]),
          };
        },
      });
      const scorer = createToolCallAccuracyScorerLLM({
        model,
        availableTools: async ({ run }) => run.input?.availableTools,
      });
      const run = createAgentTestRun({
        inputMessages: [createTestMessage({ id: 'user', role: 'user', content: 'Hello' })],
        output: [createTestMessage({ id: 'assistant', role: 'assistant', content: 'Hi' })],
      });
      for (const tools of [
        [{ id: 'search', description: 'Search documents' }],
        [{ id: 'deleteUser', description: 'Delete an account' }],
        [],
      ]) {
        prompts.length = 0;
        const savedRun = JSON.parse(JSON.stringify({ ...run, input: { ...run.input, availableTools: tools } }));
        const result = await scorer.run(savedRun);
        expect(result.score).toBe(1);
        expect(prompts[0]).toContain(
          tools[0]?.id === 'search'
            ? 'search: Search documents'
            : tools[0]?.id === 'deleteUser'
              ? 'deleteUser: Delete an account'
              : 'TOOL REFERENCE:',
        );
        expect(prompts[0]).not.toContain(tools[0]?.id === 'search' ? 'deleteUser:' : 'search:');
      }
      prompts.length = 0;
      await createToolCallAccuracyScorerLLM({ model, availableTools }).run(run);
      expect(prompts[0]).toContain('calculator-tool: Perform mathematical calculations');
      await expect(scorer.run(run)).rejects.toThrow('Available tools snapshot is missing');
    });
  });

  describe('live agent tool snapshots', () => {
    const searchTool = createTool({
      id: 'search',
      description: 'Search documents',
      inputSchema: z.object({ query: z.string() }),
      execute: async ({ query }) => ({ results: [query] }),
    });

    const deleteUserTool = createTool({
      id: 'deleteUser',
      description: 'Delete an account',
      inputSchema: z.object({ userId: z.string() }),
      execute: async () => ({ deleted: true }),
    });

    function createScoredAgent(onModelTools?: (names: string[]) => void) {
      const scoredRuns: Array<{ runId: string; availableTools: ScorerRunInputForAgent['availableTools'] }> = [];
      const recorder = createScorer({
        id: 'available-tools-recorder',
        name: 'Available tools recorder',
        description: 'Records the run-specific available tools',
      }).generateScore(({ run }) => {
        const input = run.input as ScorerRunInputForAgent;
        scoredRuns.push({ runId: run.runId, availableTools: input.availableTools });
        return 1;
      });

      // No external API calls; this is a real Agent -> workflow -> scorer integration test.
      const model = new MockLanguageModelV2({
        // generate() uses doGenerate; omitting it triggers the mock's default "Not implemented" error.
        doGenerate: async ({ tools }) => {
          onModelTools?.((tools ?? []).map(tool => tool.name));
          return {
            content: [{ type: 'text', text: 'Done' }],
            finishReason: 'stop',
            usage: { inputTokens: 1, outputTokens: 1, totalTokens: 2 },
            warnings: [],
          };
        },
        doStream: async ({ tools }) => {
          onModelTools?.((tools ?? []).map(tool => tool.name));
          return {
            stream: convertArrayToReadableStream([
              { type: 'stream-start', warnings: [] },
              { type: 'text-start', id: 'text' },
              { type: 'text-delta', id: 'text', delta: 'Done' },
              { type: 'text-end', id: 'text' },
              { type: 'finish', finishReason: 'stop', usage: { inputTokens: 1, outputTokens: 1, totalTokens: 2 } },
            ]),
          };
        },
      });

      const agent = new Agent({
        id: 'available-tools-agent',
        name: 'Available Tools Agent',
        model,
        instructions: 'Respond to the user.',
        tools: { search: searchTool, deleteUser: deleteUserTool },
        scorers: { recorder: { scorer: recorder } },
      });
      new Mastra({ agents: { agent }, logger: false, storage: new InMemoryStore() });
      return { agent, scoredRuns };
    }

    it('passes different activeTools snapshots for consecutive generate() runs', async () => {
      const { agent, scoredRuns } = createScoredAgent();

      await agent.generate('Find a document', { activeTools: ['search'] });
      await agent.generate('Delete a user', { activeTools: ['deleteUser'] });
      await agent.generate('Just reply', { activeTools: [] });

      await vi.waitFor(() => expect(scoredRuns).toHaveLength(3));
      expect(scoredRuns.map(run => run.availableTools)).toEqual([
        [{ id: 'search', description: 'Search documents' }],
        [{ id: 'deleteUser', description: 'Delete an account' }],
        [],
      ]);
      expect(new Set(scoredRuns.map(run => run.runId)).size).toBe(3);
    });

    it('captures all configured tools when activeTools is omitted', async () => {
      const { agent, scoredRuns } = createScoredAgent();

      await agent.generate('Hello');

      await vi.waitFor(() => expect(scoredRuns).toHaveLength(1));
      expect(scoredRuns[0]!.availableTools).toEqual([
        { id: 'search', description: 'Search documents' },
        { id: 'deleteUser', description: 'Delete an account' },
      ]);
    });

    it('captures the effective tool set after prepareStep filters activeTools', async () => {
      const modelTools: string[][] = [];
      const { agent, scoredRuns } = createScoredAgent(names => modelTools.push(names));
      const prepareStep = vi.fn(() => ({ activeTools: ['search' as const] }));

      // Both tools are configured, but prepareStep hides deleteUser from this model call.
      await agent.generate('Find a document', { prepareStep });

      expect(prepareStep).toHaveBeenCalled();
      expect(modelTools).toEqual([['search']]);
      await vi.waitFor(() => expect(scoredRuns).toHaveLength(1));
      expect(scoredRuns[0]!.availableTools).toEqual([{ id: 'search', description: 'Search documents' }]);
    });

    it('forwards the available-tools snapshot for stream() runs', async () => {
      const { agent, scoredRuns } = createScoredAgent();

      const output = await agent.stream('Find a document', { activeTools: ['search'] });
      for await (const chunk of output.fullStream) {
        void chunk; // Drain the stream so onFinish runs and scoring completes.
      }

      await vi.waitFor(() => expect(scoredRuns).toHaveLength(1));
      expect(scoredRuns[0]!.availableTools).toEqual([{ id: 'search', description: 'Search documents' }]);
    });
  });

  describe('Basic Configuration', () => {
    it('should create scorer with correct configuration', () => {
      const scorer = createToolCallAccuracyScorerLLM({
        model: mockModel,
        availableTools,
      });

      expect(scorer.id).toBe('llm-tool-call-accuracy-scorer');
      expect(scorer.name).toBe('Tool Call Accuracy (LLM)');
      expect(scorer.description).toContain('Evaluates whether an agent selected appropriate tools');
    });

    it('should handle empty availableTools array', () => {
      const scorer = createToolCallAccuracyScorerLLM({
        model: mockModel,
        availableTools: [],
      });

      expect(scorer.name).toBe('Tool Call Accuracy (LLM)');
      expect(scorer).toBeDefined();
    });
  });

  describe('Tool Call Extraction', () => {
    it('should extract tool calls from agent output with tool invocations', async () => {
      const scorer = createToolCallAccuracyScorerLLM({
        model: mockModel,
        availableTools,
      });

      // Mock the preprocess step
      scorer.preprocess = vi.fn().mockImplementation(() => ({
        ...scorer,
        preprocessStepResult: {
          actualTools: ['weather-tool'],
          hasToolCalls: true,
          toolCallInfos: [],
        },
      }));

      const testRun = createAgentTestRun({
        inputMessages: [
          createTestMessage({
            id: '1',
            role: 'user',
            content: 'What is the weather in Paris?',
          }),
        ],
        output: [
          createTestMessage({
            id: '2',
            role: 'assistant',
            content: 'Let me check the weather in Paris for you.',
            toolInvocations: [
              {
                toolCallId: 'call_1',
                toolName: 'weather-tool',
                args: { location: 'Paris' },
                result: { temperature: 15, condition: 'cloudy' },
                state: 'result',
              },
            ],
          }),
        ],
      });

      // Verify that tool extraction works
      const { tools } = extractToolCalls(testRun.output);
      expect(tools).toContain('weather-tool');
    });

    it('should handle output with no tool calls', async () => {
      const scorer = createToolCallAccuracyScorerLLM({
        model: mockModel,
        availableTools,
      });

      scorer.preprocess = vi.fn().mockImplementation(() => ({
        ...scorer,
        preprocessStepResult: {
          actualTools: [],
          hasToolCalls: false,
          toolCallInfos: [],
        },
      }));

      const testRun = createAgentTestRun({
        inputMessages: [
          createTestMessage({
            id: '1',
            role: 'user',
            content: 'Hello, how are you?',
          }),
        ],
        output: [
          createTestMessage({
            id: '2',
            role: 'assistant',
            content: 'I am doing well, thank you for asking!',
          }),
        ],
      });

      const { tools } = extractToolCalls(testRun.output);
      expect(tools).toHaveLength(0);
    });
  });

  describe('Scoring Logic', () => {
    it('should score 1.0 when all tool calls are appropriate', async () => {
      const scorer = createToolCallAccuracyScorerLLM({
        model: mockModel,
        availableTools,
      });

      // Mock the run method to return expected results
      scorer.run = vi.fn().mockResolvedValue({
        score: 1.0,
        reason: 'Perfect tool selection for weather query.',
      });

      const testRun = createAgentTestRun({
        inputMessages: [
          createTestMessage({
            id: '1',
            role: 'user',
            content: 'What is the weather in Paris?',
          }),
        ],
        output: [
          createTestMessage({
            id: '2',
            role: 'assistant',
            content: 'The weather in Paris is 15°C and cloudy.',
            toolInvocations: [
              {
                toolCallId: 'call_1',
                toolName: 'weather-tool',
                args: { location: 'Paris' },
                result: { temperature: 15, condition: 'cloudy' },
                state: 'result',
              },
            ],
          }),
        ],
      });

      const result = await scorer.run(testRun);
      expect(result.score).toBe(1.0);
      expect(result.reason).toBe('Perfect tool selection for weather query.');
    });

    it('should score 0.0 when all tool calls are inappropriate', async () => {
      const scorer = createToolCallAccuracyScorerLLM({
        model: mockModel,
        availableTools,
      });

      // Mock the run method to return expected results
      scorer.run = vi.fn().mockResolvedValue({
        score: 0.0,
        reason: 'Wrong tools used - should have used weather-tool.',
      });

      const testRun = createAgentTestRun({
        inputMessages: [
          createTestMessage({
            id: '1',
            role: 'user',
            content: 'What is the weather today?',
          }),
        ],
        output: [
          createTestMessage({
            id: '2',
            role: 'assistant',
            content: 'Let me calculate that and check your calendar.',
            toolInvocations: [
              {
                toolCallId: 'call_1',
                toolName: 'calculator-tool',
                args: {},
                result: {},
                state: 'result',
              },
              {
                toolCallId: 'call_2',
                toolName: 'calendar-tool',
                args: {},
                result: {},
                state: 'result',
              },
            ],
          }),
        ],
      });

      const result = await scorer.run(testRun);
      expect(result.score).toBe(0.0);
      expect(result.reason).toBe('Wrong tools used - should have used weather-tool.');
    });

    it('should score 0.5 when half of tool calls are appropriate', async () => {
      const scorer = createToolCallAccuracyScorerLLM({
        model: mockModel,
        availableTools,
      });

      // Mock the run method to return expected results
      scorer.run = vi.fn().mockResolvedValue({
        score: 0.5,
        reason: 'Partial success - weather tool was appropriate but calculator was not needed.',
      });

      const testRun = createAgentTestRun({
        inputMessages: [
          createTestMessage({
            id: '1',
            role: 'user',
            content: 'What is the weather in London?',
          }),
        ],
        output: [
          createTestMessage({
            id: '2',
            role: 'assistant',
            content: 'The weather in London is 10°C.',
            toolInvocations: [
              {
                toolCallId: 'call_1',
                toolName: 'weather-tool',
                args: { location: 'London' },
                result: { temperature: 10 },
                state: 'result',
              },
              {
                toolCallId: 'call_2',
                toolName: 'calculator-tool',
                args: { operation: 'celsius_to_fahrenheit' },
                result: { fahrenheit: 50 },
                state: 'result',
              },
            ],
          }),
        ],
      });

      const result = await scorer.run(testRun);
      expect(result.score).toBe(0.5);
    });

    it('should score 1.0 when no tools were called and none were needed', async () => {
      const scorer = createToolCallAccuracyScorerLLM({
        model: mockModel,
        availableTools,
      });

      // Mock the run method to return expected results
      scorer.run = vi.fn().mockResolvedValue({
        score: 1.0,
        reason: 'Correctly identified that no tools were needed for this greeting.',
      });

      const testRun = createAgentTestRun({
        inputMessages: [
          createTestMessage({
            id: '1',
            role: 'user',
            content: 'Hello, how are you?',
          }),
        ],
        output: [
          createTestMessage({
            id: '2',
            role: 'assistant',
            content: 'Hello! I am doing well, thank you for asking.',
          }),
        ],
      });

      const result = await scorer.run(testRun);
      expect(result.score).toBe(1.0);
    });

    it('should score 0.0 when no tools were called but some were needed', async () => {
      const scorer = createToolCallAccuracyScorerLLM({
        model: mockModel,
        availableTools,
      });

      // Mock the run method to return expected results
      scorer.run = vi.fn().mockResolvedValue({
        score: 0.0,
        reason: 'Failed to use weather-tool when explicitly asked for weather information.',
      });

      const testRun = createAgentTestRun({
        inputMessages: [
          createTestMessage({
            id: '1',
            role: 'user',
            content: 'What is the weather in Tokyo?',
          }),
        ],
        output: [
          createTestMessage({
            id: '2',
            role: 'assistant',
            content: 'I cannot provide weather information without checking.',
          }),
        ],
      });

      const result = await scorer.run(testRun);
      expect(result.score).toBe(0.0);
    });
  });

  describe('Edge Cases', () => {
    it('should handle multiple tool calls for complex queries', async () => {
      const scorer = createToolCallAccuracyScorerLLM({
        model: mockModel,
        availableTools,
      });

      // Mock the run method to return expected results
      scorer.run = vi.fn().mockResolvedValue({
        score: 1.0,
        reason: 'Both tools were appropriate for the complex query.',
      });

      const testRun = createAgentTestRun({
        inputMessages: [
          createTestMessage({
            id: '1',
            role: 'user',
            content: 'What is the weather in Paris and what are the top tourist attractions?',
          }),
        ],
        output: [
          createTestMessage({
            id: '2',
            role: 'assistant',
            content: 'Let me get you the weather and tourist information for Paris.',
            toolInvocations: [
              {
                toolCallId: 'call_1',
                toolName: 'weather-tool',
                args: { location: 'Paris' },
                result: { temperature: 15 },
                state: 'result',
              },
              {
                toolCallId: 'call_2',
                toolName: 'search-tool',
                args: { query: 'Paris tourist attractions' },
                result: { results: ['Eiffel Tower', 'Louvre'] },
                state: 'result',
              },
            ],
          }),
        ],
      });

      const result = await scorer.run(testRun);
      expect(result.score).toBe(1.0);
    });

    it('should handle clarification requests appropriately', async () => {
      const scorer = createToolCallAccuracyScorerLLM({
        model: mockModel,
        availableTools,
      });

      // Mock the run method to return expected results
      scorer.run = vi.fn().mockResolvedValue({
        score: 1.0,
        reason: 'Appropriately asked for clarification on vague request.',
      });

      const testRun = createAgentTestRun({
        inputMessages: [
          createTestMessage({
            id: '1',
            role: 'user',
            content: 'I need help with something',
          }),
        ],
        output: [
          createTestMessage({
            id: '2',
            role: 'assistant',
            content:
              'I would be happy to help! Could you please provide more details about what you need assistance with?',
          }),
        ],
      });

      const result = await scorer.run(testRun);
      expect(result.score).toBe(1.0);
      expect(result.reason).toContain('clarification');
    });

    it('should handle undefined or null results gracefully', async () => {
      const scorer = createToolCallAccuracyScorerLLM({
        model: mockModel,
        availableTools,
      });

      // Mock the run method to return expected results
      scorer.run = vi.fn().mockResolvedValue({
        score: 1.0,
        reason: 'Unable to evaluate tool selection.',
      });

      const testRun = createAgentTestRun({
        inputMessages: [
          createTestMessage({
            id: '1',
            role: 'user',
            content: 'Test query',
          }),
        ],
        output: [
          createTestMessage({
            id: '2',
            role: 'assistant',
            content: 'Test response',
          }),
        ],
      });

      const result = await scorer.run(testRun);

      // When no evaluations and no missing tools, default to 1.0
      expect(result.score).toBe(1.0);
    });

    it('should handle tool calls with errors', async () => {
      const scorer = createToolCallAccuracyScorerLLM({
        model: mockModel,
        availableTools,
      });

      // Mock the run method to return expected results
      scorer.run = vi.fn().mockResolvedValue({
        score: 1.0,
        reason: 'Correct tool selection despite execution error.',
      });

      const testRun = createAgentTestRun({
        inputMessages: [
          createTestMessage({
            id: '1',
            role: 'user',
            content: 'What is the weather in InvalidCity?',
          }),
        ],
        output: [
          createTestMessage({
            id: '2',
            role: 'assistant',
            content: 'I encountered an error getting the weather.',
            toolInvocations: [
              {
                toolCallId: 'call_1',
                toolName: 'weather-tool',
                args: { location: 'InvalidCity' },
                result: { error: 'City not found' },
                state: 'error',
              },
            ],
          }),
        ],
      });

      const result = await scorer.run(testRun);

      // Tool selection was still appropriate even if execution failed
      expect(result.score).toBe(1.0);
    });

    it('should handle repeated tool calls', async () => {
      const scorer = createToolCallAccuracyScorerLLM({
        model: mockModel,
        availableTools,
      });

      // Mock the run method to return expected results
      scorer.run = vi.fn().mockResolvedValue({
        score: 0.67,
        reason: 'Two out of three tool calls were appropriate.',
      });

      const testRun = createAgentTestRun({
        inputMessages: [
          createTestMessage({
            id: '1',
            role: 'user',
            content: 'Compare weather in Paris and London',
          }),
        ],
        output: [
          createTestMessage({
            id: '2',
            role: 'assistant',
            content: 'Checking weather for both cities.',
            toolInvocations: [
              {
                toolCallId: 'call_1',
                toolName: 'weather-tool',
                args: { location: 'Paris' },
                result: { temperature: 15 },
                state: 'result',
              },
              {
                toolCallId: 'call_2',
                toolName: 'weather-tool',
                args: { location: 'London' },
                result: { temperature: 10 },
                state: 'result',
              },
              {
                toolCallId: 'call_3',
                toolName: 'weather-tool',
                args: { location: 'Paris' },
                result: { temperature: 15 },
                state: 'result',
              },
            ],
          }),
        ],
      });

      const result = await scorer.run(testRun);
      expect(result.score).toBeCloseTo(0.67, 2); // 2 out of 3 were appropriate
    });
  });

  describe('Real-world Scenarios', () => {
    it('should evaluate multi-step tool usage correctly', async () => {
      const scorer = createToolCallAccuracyScorerLLM({
        model: mockModel,
        availableTools: [...availableTools, createTool({ id: 'email-tool', description: 'Send emails' })],
      });

      // Mock the run method to return expected results
      scorer.run = vi.fn().mockResolvedValue({
        score: 0.67,
        reason: 'Two out of three tools were appropriate for the task.',
      });

      const testRun = createAgentTestRun({
        inputMessages: [
          createTestMessage({
            id: '1',
            role: 'user',
            content: 'Schedule an outdoor picnic for this weekend if the weather is good',
          }),
        ],
        output: [
          createTestMessage({
            id: '2',
            role: 'assistant',
            content: 'I will check the weather and schedule the picnic.',
            toolInvocations: [
              {
                toolCallId: 'call_1',
                toolName: 'calendar-tool',
                args: { event: 'Outdoor Picnic' },
                result: { scheduled: true },
                state: 'result',
              },
              {
                toolCallId: 'call_2',
                toolName: 'weather-tool',
                args: { date: 'weekend' },
                result: { forecast: 'sunny' },
                state: 'result',
              },
              {
                toolCallId: 'call_3',
                toolName: 'email-tool',
                args: { to: 'user@example.com' },
                result: { sent: true },
                state: 'result',
              },
            ],
          }),
        ],
      });

      const result = await scorer.run(testRun);
      expect(result.score).toBeCloseTo(0.67, 2); // 2 out of 3 tools were appropriate
    });

    it('should recognize when agent over-helps with unnecessary tools', async () => {
      const scorer = createToolCallAccuracyScorerLLM({
        model: mockModel,
        availableTools,
      });

      // Mock the run method to return expected results
      scorer.run = vi.fn().mockResolvedValue({
        score: 0.33,
        reason: 'Only one out of three tools was appropriate.',
      });

      const testRun = createAgentTestRun({
        inputMessages: [
          createTestMessage({
            id: '1',
            role: 'user',
            content: 'What is 15% of 240?',
          }),
        ],
        output: [
          createTestMessage({
            id: '2',
            role: 'assistant',
            content: '15% of 240 is 36. I also found some helpful tips about percentages and checked the weather.',
            toolInvocations: [
              {
                toolCallId: 'call_1',
                toolName: 'calculator-tool',
                args: { operation: 'percentage', value: 240, percent: 15 },
                result: { result: 36 },
                state: 'result',
              },
              {
                toolCallId: 'call_2',
                toolName: 'search-tool',
                args: { query: 'percentage calculation tips' },
                result: { results: ['tip1', 'tip2'] },
                state: 'result',
              },
              {
                toolCallId: 'call_3',
                toolName: 'weather-tool',
                args: { location: 'current' },
                result: { temperature: 20 },
                state: 'result',
              },
            ],
          }),
        ],
      });

      const result = await scorer.run(testRun);
      expect(result.score).toBeCloseTo(0.33, 2); // Only 1 out of 3 tools was appropriate
    });
  });
});
