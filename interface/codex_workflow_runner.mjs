#!/usr/bin/env node
/**
 * GEAK Workflow compatibility runtime for Codex CLI.
 *
 * Claude Code's Workflow feature evaluates the repository's JavaScript files
 * with a handful of injected globals.  This runner provides the same small
 * surface on top of `codex exec`, allowing the workflow sources to stay
 * provider-neutral and preserving nested workflows and concurrency.
 *
 * Authentication is deliberately not handled here.  Each `codex exec` child
 * inherits CODEX_HOME and the current environment, so `codex login` (including
 * ChatGPT subscription authentication) works exactly as it does at the shell.
 */

import { spawn } from "node:child_process";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;

function env(name, fallback = "") {
  const value = process.env[name];
  return value == null || value === "" ? fallback : value;
}

function positiveInt(name, fallback) {
  const value = Number.parseInt(env(name, String(fallback)), 10);
  return Number.isFinite(value) && value > 0 ? value : fallback;
}

const config = {
  codexBin: env("GEAK_CODEX_BIN", "codex"),
  model: env("GEAK_CODEX_MODEL", "gpt-5.6-sol"),
  reasoningEffort: env("GEAK_CODEX_REASONING_EFFORT", "high"),
  sandbox: env("GEAK_CODEX_SANDBOX", "danger-full-access"),
  concurrency: positiveInt("GEAK_CODEX_CONCURRENCY", 8),
  cwd: resolve(env("GEAK_CODEX_CWD", process.cwd())),
  ephemeral: env("GEAK_CODEX_EPHEMERAL", "1") !== "0",
  bypassApprovals: env("GEAK_CODEX_BYPASS_APPROVALS", "1") !== "0",
};

let activeAgents = 0;
const agentWaiters = [];

async function acquireAgentSlot() {
  if (activeAgents < config.concurrency) {
    activeAgents += 1;
    return;
  }
  await new Promise((resolveWaiter) => agentWaiters.push(resolveWaiter));
}

function releaseAgentSlot() {
  const next = agentWaiters.shift();
  if (next) next(); // transfer this slot directly to the oldest waiter
  else activeAgents -= 1;
}

function containsOpenMap(node) {
  if (!node || typeof node !== "object") return false;
  if (node.type === "object" && node.additionalProperties === true &&
      Object.keys(node.properties || {}).length === 0) return true;
  if (node.properties && Object.values(node.properties).some(containsOpenMap)) return true;
  if (node.items && containsOpenMap(node.items)) return true;
  for (const key of ["anyOf", "oneOf", "allOf"]) {
    if (Array.isArray(node[key]) && node[key].some(containsOpenMap)) return true;
  }
  return false;
}

function nullableType(type) {
  if (Array.isArray(type)) return type.includes("null") ? type : [...type, "null"];
  return type === "null" ? type : [type, "null"];
}

function strictifySchema(node, optional = false) {
  if (!node || typeof node !== "object") return node;
  if (Array.isArray(node)) return node.map((item) => strictifySchema(item));
  const out = { ...node };
  if (out.properties && typeof out.properties === "object") {
    const originallyRequired = new Set(Array.isArray(out.required) ? out.required : []);
    out.properties = Object.fromEntries(Object.entries(out.properties).map(([key, value]) => [
      key, strictifySchema(value, !originallyRequired.has(key)),
    ]));
    out.required = Object.keys(out.properties);
  }
  if (out.items) out.items = strictifySchema(out.items);
  for (const key of ["anyOf", "oneOf", "allOf"]) {
    if (Array.isArray(out[key])) out[key] = out[key].map((item) => strictifySchema(item));
  }
  if (out.type === "object" || (Array.isArray(out.type) && out.type.includes("object"))) {
    out.additionalProperties = false;
  }
  if (optional && out.type) out.type = nullableType(out.type);
  return out;
}

function normalizeSchema(schema) {
  const value = schema && typeof schema === "object"
    ? schema : { type: "object", properties: {}, required: [], additionalProperties: false };
  return {
    schema: containsOpenMap(value) ? null : strictifySchema(value),
    promptSchema: value,
  };
}

function parseJsonObject(text, label) {
  const value = String(text || "").trim();
  try {
    const parsed = JSON.parse(value);
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) return parsed;
  } catch (_) {
    // Fall through to a balanced-object scan for defensive compatibility with
    // older Codex builds that may wrap the final value in prose or a fence.
  }

  let start = -1;
  let depth = 0;
  let inString = false;
  let escaped = false;
  let last = null;
  for (let i = 0; i < value.length; i += 1) {
    const ch = value[i];
    if (inString) {
      if (escaped) escaped = false;
      else if (ch === "\\") escaped = true;
      else if (ch === '"') inString = false;
      continue;
    }
    if (ch === '"') inString = true;
    else if (ch === "{") {
      if (depth === 0) start = i;
      depth += 1;
    } else if (ch === "}" && depth > 0) {
      depth -= 1;
      if (depth === 0 && start >= 0) {
        try {
          const candidate = JSON.parse(value.slice(start, i + 1));
          if (candidate && typeof candidate === "object" && !Array.isArray(candidate)) last = candidate;
        } catch (_) {
          // Continue looking for a later valid object.
        }
        start = -1;
      }
    }
  }
  if (last) return last;
  throw new Error(`${label} returned no JSON object: ${value.slice(-1000)}`);
}

async function runProcess(command, args, { cwd, stdin = "", label }) {
  return await new Promise((resolveRun, rejectRun) => {
    const child = spawn(command, args, {
      cwd,
      env: process.env,
      stdio: ["pipe", "pipe", "pipe"],
    });
    const stdout = [];
    const stderr = [];
    child.stdout.on("data", (chunk) => stdout.push(chunk));
    child.stderr.on("data", (chunk) => {
      stderr.push(chunk);
      if (env("GEAK_CODEX_STREAM_PROGRESS", "1") !== "0") process.stderr.write(chunk);
    });
    child.on("error", rejectRun);
    child.on("close", (code, signal) => {
      const out = Buffer.concat(stdout).toString("utf8");
      const err = Buffer.concat(stderr).toString("utf8");
      if (code === 0) resolveRun({ stdout: out, stderr: err });
      else rejectRun(new Error(`${label} failed (code=${code}, signal=${signal || "none"}): ${err.slice(-3000)}`));
    });
    child.stdin.end(stdin);
  });
}

async function codexAgent(prompt, options = {}) {
  await acquireAgentSlot();
  const scratch = await mkdtemp(`${tmpdir()}/geak-codex-agent-`);
  const schemaPath = resolve(scratch, "schema.json");
  const outputPath = resolve(scratch, "output.json");
  const label = options.label || "agent";
  try {
    const normalized = normalizeSchema(options.schema);
    if (normalized.schema) {
      await writeFile(schemaPath, JSON.stringify(normalized.schema), "utf8");
    }
    const args = ["exec", "-", "--cd", config.cwd, "--skip-git-repo-check",
      "--model", config.model,
      "--config", `model_reasoning_effort=${JSON.stringify(options.effort || config.reasoningEffort)}`,
      "--output-last-message", outputPath,
      "--color", "never"];
    if (normalized.schema) args.push("--output-schema", schemaPath);
    if (config.ephemeral) args.push("--ephemeral");
    if (config.bypassApprovals) args.push("--dangerously-bypass-approvals-and-sandbox");
    else args.push("--sandbox", config.sandbox);

    const phase = options.phase ? `GEAK phase: ${options.phase}\n` : "";
    const instruction =
      `${phase}GEAK agent label: ${label}\n` +
      "Work directly in the provided environment. Follow the task exactly. " +
      "Your final response must be only the JSON object required by this schema:\n" +
      `${JSON.stringify(normalized.promptSchema)}\n\n` +
      String(prompt);
    const result = await runProcess(config.codexBin, args, {
      cwd: config.cwd,
      stdin: instruction,
      label: `codex agent ${label}`,
    });
    let output = result.stdout;
    try {
      output = await readFile(outputPath, "utf8");
    } catch (_) {
      // stdout is the documented final-message channel and is a valid fallback.
    }
    return parseJsonObject(output, `codex agent ${label}`);
  } finally {
    releaseAgentSlot();
    await rm(scratch, { recursive: true, force: true });
  }
}

async function parallel(thunks) {
  return await Promise.all((thunks || []).map((thunk) => thunk()));
}

async function pipeline(items, ...stages) {
  return await Promise.all((items || []).map(async (item, index) => {
    let value = item;
    for (const stage of stages) value = await stage(value, item, index);
    return value;
  }));
}

async function executeWorkflow(scriptPath, args) {
  const absoluteScript = resolve(scriptPath);
  const source = await readFile(absoluteScript, "utf8");
  // `export const meta` is metadata consumed by Claude's discovery layer.  The
  // workflow body only needs a local binding when evaluated by this runtime.
  const compatibleSource = source.replace(/^export\s+const\s+meta\s*=/m, "const meta =");
  const phase = (name) => process.stderr.write(`[geak:${name}]\n`);
  const log = (message) => process.stderr.write(`[geak] ${String(message)}\n`);
  const workflow = async (reference, childArgs) => {
    if (!reference || !reference.scriptPath) throw new Error("workflow() requires reference.scriptPath");
    return await executeWorkflow(reference.scriptPath, childArgs || {});
  };
  const fn = new AsyncFunction(
    "args", "agent", "workflow", "parallel", "pipeline", "phase", "log",
    `"use strict";\n${compatibleSource}\n`,
  );
  return await fn(args || {}, codexAgent, workflow, parallel, pipeline, phase, log);
}

function usage() {
  return "usage: codex_workflow_runner.mjs --script PATH --args-file PATH [--smoke-agent]";
}

async function main(argv) {
  let script = "";
  let argsFile = "";
  let smokeAgent = false;
  for (let i = 0; i < argv.length; i += 1) {
    if (argv[i] === "--script") script = argv[++i] || "";
    else if (argv[i] === "--args-file") argsFile = argv[++i] || "";
    else if (argv[i] === "--smoke-agent") smokeAgent = true;
    else throw new Error(`unknown argument: ${argv[i]}\n${usage()}`);
  }
  if (smokeAgent) {
    const result = await codexAgent("Return {\"status\":\"ok\",\"backend\":\"codex\"}.", {
      phase: "Smoke", label: "runtime-smoke",
      schema: {
        type: "object",
        properties: {
          status: { type: "string" },
          backend: { type: "string" },
          note: { type: "string" },
        },
        required: ["status", "backend"],
        additionalProperties: true,
      },
    });
    process.stdout.write(`${JSON.stringify(result)}\n`);
    return;
  }
  if (!script || !argsFile) throw new Error(usage());
  const workflowArgs = JSON.parse(await readFile(resolve(argsFile), "utf8"));
  const result = await executeWorkflow(script, workflowArgs);
  process.stdout.write(`${JSON.stringify(result)}\n`);
}

if (resolve(process.argv[1] || "") === fileURLToPath(import.meta.url)) {
  main(process.argv.slice(2)).catch((error) => {
    process.stderr.write(`GEAK Codex workflow failed: ${error && error.stack ? error.stack : error}\n`);
    process.exitCode = 1;
  });
}

export { codexAgent, executeWorkflow, normalizeSchema, parallel, pipeline };
