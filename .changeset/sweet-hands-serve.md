---
'@mastra/core': minor
---

Added initial per-run tool snapshots to agent scorer inputs and trace metadata for delayed scoring. Scorers can read run.input.availableTools; older runs keep this optional.

```ts
createScorer({ id: 'tool-count', type: 'agent' }).generateScore(({ run }) => {
  return run.input?.availableTools?.length ?? 0;
});
```
