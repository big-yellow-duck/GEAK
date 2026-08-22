import assert from "node:assert/strict";
import { chmod, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import test from "node:test";

const RUNNER = new URL("./codex_workflow_runner.mjs", import.meta.url);

test("Codex runtime implements agent, nested workflow, parallel, and pipeline", async () => {
  const root = await mkdtemp(`${tmpdir()}/geak-codex-runtime-test-`);
  const mockCodex = resolve(root, "codex");
  const child = resolve(root, "child.js");
  const parent = resolve(root, "parent.js");
  try {
    await writeFile(mockCodex, `#!/bin/sh
set -eu
out=""
config=""
while [ "$#" -gt 0 ]; do
  if [ "$1" = "--output-last-message" ]; then out="$2"; shift 2
  elif [ "$1" = "--config" ]; then config="$2"; shift 2
  else shift
  fi
done
prompt=$(cat)
case "$prompt" in
  *child-agent*) [ "$config" = 'model_reasoning_effort="low"' ] || exit 9; value=11 ;;
  *parallel-a*) value=20 ;;
  *parallel-b*) value=21 ;;
  *pipeline-agent*) value=30 ;;
  *) value=0 ;;
esac
printf '{"value":%s}\n' "$value" > "$out"
printf '{"value":%s}\n' "$value"
`, "utf8");
    await chmod(mockCodex, 0o755);
    await writeFile(child, `
export const meta = {name: 'child'};
phase('Child');
const result = await agent('child-agent', {label: 'child', effort: 'low', schema: {type: 'object'}});
return {child: result.value, seed: args.seed};
`, "utf8");
    await writeFile(parent, `
export const meta = {name: 'parent'};
phase('Parent');
const nested = await workflow({scriptPath: args.child}, {seed: 7});
const both = await parallel([
  () => agent('parallel-a', {label: 'a', schema: {type: 'object'}}),
  () => agent('parallel-b', {label: 'b', schema: {type: 'object'}}),
]);
const piped = await pipeline([5],
  async (value) => ({value}),
  async (item) => ({input: item.value, result: await agent('pipeline-agent', {schema: {type: 'object'}})}));
log('done');
return {nested, both: both.map(x => x.value), piped: piped[0]};
`, "utf8");

    process.env.GEAK_CODEX_BIN = mockCodex;
    process.env.GEAK_CODEX_CWD = root;
    process.env.GEAK_CODEX_STREAM_PROGRESS = "0";
    process.env.GEAK_CODEX_CONCURRENCY = "2";
    const { executeWorkflow } = await import(RUNNER);
    const result = await executeWorkflow(parent, { child });
    assert.deepEqual(result, {
      nested: { child: 11, seed: 7 },
      both: [20, 21],
      piped: { input: 5, result: { value: 30 } },
    });
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("Claude-style schemas are adapted without flattening open maps", async () => {
  const { normalizeSchema } = await import(RUNNER);
  const closed = normalizeSchema({
    type: "object",
    properties: {
      required_value: { type: "string" },
      optional_value: { type: "number" },
      nested: {
        type: "object",
        properties: { x: { type: "boolean" } },
        required: ["x"],
        additionalProperties: true,
      },
    },
    required: ["required_value"],
    additionalProperties: true,
  });
  assert.equal(closed.schema.additionalProperties, false);
  assert.deepEqual(closed.schema.required, ["required_value", "optional_value", "nested"]);
  assert.deepEqual(closed.schema.properties.optional_value.type, ["number", "null"]);
  assert.equal(closed.schema.properties.nested.additionalProperties, false);

  const open = normalizeSchema({
    type: "object",
    properties: { metadata: { type: "object", additionalProperties: true } },
    required: ["metadata"],
    additionalProperties: true,
  });
  assert.equal(open.schema, null);
  assert.equal(open.promptSchema.properties.metadata.additionalProperties, true);
});
