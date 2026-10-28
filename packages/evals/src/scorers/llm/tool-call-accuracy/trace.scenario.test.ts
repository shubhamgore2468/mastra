import { createOpenAI } from '@ai-sdk/openai-v6';
import { LLMock } from '@copilotkit/aimock';
import { Agent } from '@mastra/core/agent';
import type { ToolsInput } from '@mastra/core/agent';
import { scoreTrace } from '@mastra/core/evals/scoreTraces';
import { Mastra } from '@mastra/core/mastra';
import { SpanType } from '@mastra/core/observability';
import { RequestContext } from '@mastra/core/request-context';
import { InMemoryStore } from '@mastra/core/storage';
import { createTool } from '@mastra/core/tools';
import { afterAll, afterEach, beforeAll, describe, expect, it } from 'vitest';
import { z } from 'zod/v4';
// Exercise the real exporter without requiring a published observability build.
import { Observability } from '../../../../../../observability/mastra/src/default';
import { MastraStorageExporter } from '../../../../../../observability/mastra/src/exporters/mastra-storage';
import { createToolCallAccuracyScorerLLM } from './index';

describe.each(['generate', 'stream'] as const)('Tool Call Accuracy from persisted Agent.%s traces', method => {
  let llm: LLMock;
  let observability: Observability | undefined;

  beforeAll(async () => {
    llm = new LLMock({ port: 0 });
    await llm.start();
  });
  afterEach(async () => {
    await observability?.shutdown();
    llm.clearFixtures();
    llm.clearRequests();
    llm.resetMatchCounts();
  });
  afterAll(async () => {
    await llm.stop();
  });

  it('scores stored traces using their original tools after the agent configuration changes', async () => {
    const model = createOpenAI({ apiKey: 'test-key', baseURL: `${llm.url}/v1` }).chat('gpt-4o-mini');
    llm.on({ endpoint: 'chat' }, { content: 'I cannot delete accounts with my available tools.' });
    const search = createTool({
      id: 'internal-search',
      description: 'Search documents',
      inputSchema: z.object({}),
      execute: async () => 'Documents',
    });
    const deleteUser = createTool({
      id: 'deleteUser',
      description: 'Delete an account',
      inputSchema: z.object({}),
      execute: async () => 'Deleted',
    });
    const storage = new InMemoryStore();
    observability = new Observability({
      configs: {
        default: { serviceName: 'tool-availability-test', exporters: [new MastraStorageExporter()] },
      },
    });
    let currentTools: ToolsInput = { search, deleteUser };
    const agent = new Agent({
      id: 'account-assistant',
      name: 'Account assistant',
      instructions: 'Help with account requests.',
      model,
      tools: ({ requestContext }): ToolsInput => (requestContext.get('role') === 'admin' ? currentTools : { search }),
    });
    new Mastra({ agents: { agent }, storage, observability, logger: false });
    const cases = [
      {
        role: 'admin',
        activeTools: undefined,
        expected: [
          { id: 'search', description: 'Search documents' },
          { id: 'deleteUser', description: 'Delete an account' },
        ],
      },
      { role: 'viewer', activeTools: undefined, expected: [{ id: 'search', description: 'Search documents' }] },
      { role: 'admin', activeTools: [], expected: [] },
    ];
    const traceIds: string[] = [];
    for (const { role, activeTools } of cases) {
      const requestContext = new RequestContext();
      requestContext.set('role', role);
      if (method === 'generate') {
        const result = await agent.generate('Delete user demo-123.', { requestContext, activeTools });
        expect(result.traceId).toBeTruthy();
        traceIds.push(result.traceId!);
      } else {
        const result = await agent.stream('Delete user demo-123.', { requestContext, activeTools });
        await result.consumeStream();
        expect(result.traceId).toBeTruthy();
        traceIds.push(result.traceId!);
      }
    }
    await observability.flush();
    await observability.shutdown();
    observability = undefined;

    // Historical scoring has only storage and a new scorer, not the agent or its old configuration.
    search.description = 'Changed after the run';
    currentTools = {};
    llm.clearFixtures();
    llm.clearRequests();
    llm.on({ endpoint: 'chat' }, { content: JSON.stringify({ evaluations: [], missingTools: [] }) });
    const scorer = createToolCallAccuracyScorerLLM({ model, availableTools: ({ run }) => run.input?.availableTools });
    const traces = (await storage.getStore('observability'))!;
    const scores = (await storage.getStore('scores'))!;

    for (const [index, { expected }] of cases.entries()) {
      const traceId = traceIds[index]!;
      const trace = await traces.getTrace({ traceId });
      expect(trace?.spans.find(span => span.spanType === SpanType.AGENT_RUN)?.metadata?.availableTools).toEqual(
        expected,
      );
      llm.clearRequests();
      const result = await scoreTrace({ storage, scorer, target: { traceId } });
      expect(result?.input.availableTools).toEqual(expected);
      expect(result?.score).toBe(1);
      const persisted = await scores.getScoreById({ id: result!.id });
      expect(persisted?.input.availableTools).toEqual(expected);
      const messages = llm.getRequests()[0]!.body.messages as { role: string; content: string }[];
      const prompt = messages.find(message => message.role === 'user')!.content;
      expect(prompt.split('TOOL REFERENCE:\n')[1]?.split('\n\nEVALUATION RULES:')[0]).toBe(
        expected.map(tool => `${tool.id}: ${tool.description}`).join('\n'),
      );
      expect(prompt).not.toContain('Changed after the run');
    }
  });
});
