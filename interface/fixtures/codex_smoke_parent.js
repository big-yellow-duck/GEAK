export const meta = {
  name: 'codex-smoke-parent',
  phases: [
    { title: 'Parent', detail: 'nested workflow, parallel agents, and pipeline' },
  ],
};

const VALUE_SCHEMA = {
  type: 'object',
  properties: { kind: { type: 'string' }, value: { type: 'number' } },
  required: ['kind', 'value'],
  additionalProperties: true,
};

phase('Parent');
const child = await workflow({ scriptPath: args.child_script }, {});
const parallelResults = await parallel([
  () => agent('Return {"kind":"parallel-a","value":11}.',
    { phase: 'Parent', label: 'smoke:parallel-a', schema: VALUE_SCHEMA }),
  () => agent('Return {"kind":"parallel-b","value":13}.',
    { phase: 'Parent', label: 'smoke:parallel-b', schema: VALUE_SCHEMA }),
]);
const piped = await pipeline(
  [child, ...parallelResults],
  async (item) => item.value,
  async (value) => value * 2,
);
log('Codex compatibility smoke complete');
return { child, parallel: parallelResults, piped };
