---
'@mastra/evals': minor
---

Added per-run tool resolvers to Tool Call Accuracy. Use availableTools: ({ run }) => run.input?.availableTools instead of a static tool array to evaluate each run against its saved tool snapshot.

```ts
// Static lists remain supported.
createToolCallAccuracyScorerLLM({ model, availableTools: tools });

// Resolve the tools captured for each run.
createToolCallAccuracyScorerLLM({
  model,
  availableTools: ({ run }) => run.input?.availableTools,
});
```
