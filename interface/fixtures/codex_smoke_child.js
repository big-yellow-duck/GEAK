export const meta = {
  name: 'codex-smoke-child',
  phases: [{ title: 'Child', detail: 'one schema-constrained agent call' }],
};

phase('Child');
const result = await agent(
  'Return {"kind":"child","value":7}.',
  {
    phase: 'Child',
    label: 'smoke:child',
    effort: 'low',
    schema: {
      type: 'object',
      properties: { kind: { type: 'string' }, value: { type: 'number' } },
      required: ['kind', 'value'],
      additionalProperties: true,
    },
  },
);
return result;
