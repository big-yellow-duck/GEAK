#!/usr/bin/env node
// Contract tests for the RDNA4 architecture routing. These are intentionally
// dependency-free so they run on CI hosts without an AMD GPU.

const fs = require('fs');
const path = require('path');
const cp = require('child_process');

const root = path.resolve(__dirname, '..');
const workflow = fs.readFileSync(path.join(root, 'kernel_workflow.js'), 'utf8');
const gpuLock = fs.readFileSync(path.join(root, 'scripts/gpu_lock.sh'), 'utf8');
const profiler = fs.readFileSync(path.join(root, 'scripts/profile_kernel.sh'), 'utf8');
const rdnaDoc = fs.readFileSync(path.join(root, 'knowledge/amd_rdna4.md'), 'utf8');
const e2e = fs.readFileSync(path.join(root, '..', 'e2e_workflow/e2e_workflow.js'), 'utf8');
const capabilityIndex = fs.readFileSync(
  path.join(root, '..', 'perf_knowledge/index/capability_index.yaml'), 'utf8');
const flydslRdna = fs.readFileSync(
  path.join(root, '..', 'perf_knowledge/languages/flydsl/rdna4.md'), 'utf8');
const scaledFlydsl = fs.readFileSync(
  path.join(root, '..', 'perf_knowledge/operators/scaled_quant_gemm/backends/flydsl.md'), 'utf8');
const expertIndex = fs.readFileSync(
  path.join(root, '..', 'perf_knowledge/expert_skills/index.yaml'), 'utf8');
const rdnaSkill = fs.readFileSync(path.join(
  root, '..', 'perf_knowledge/expert_skills/skills/flydsl_rdna4_fp8_blockscale_small_m/skill.md'), 'utf8');
const authorRole = fs.readFileSync(path.join(root, 'roles/author_engineer.md'), 'utf8');
const techLeadRole = fs.readFileSync(path.join(root, 'roles/tech_lead.md'), 'utf8');

function ok(value, message) {
  if (!value) throw new Error(`FAIL: ${message}`);
  process.stdout.write(`ok - ${message}\n`);
}

for (const [gfx, arch, wave] of [
  ['gfx1201', 'rdna4', '32'],
  ['gfx1200', 'rdna4', '32'],
  ['gfx942', 'cdna3', '64'],
  ['gfx950', 'cdna4', '64'],
]) {
  const out = cp.execFileSync('bash', [path.join(root, 'scripts/detect_gpu_arch.sh')], {
    env: { ...process.env, GEAK_GPU_GFX: gfx }, encoding: 'utf8',
  });
  ok(out.includes(`GEAK_GPU_GFX=${gfx}`), `${gfx} is preserved`);
  ok(out.includes(`GEAK_GPU_ARCH_CLASS=${arch}`), `${gfx} -> ${arch}`);
  ok(out.includes(`GEAK_GPU_WAVE_SIZE=${wave}`), `${gfx} -> wave${wave}`);
}
const counts = cp.execFileSync('bash', [path.join(root, 'scripts/detect_gpu_arch.sh')], {
  env: { ...process.env, GEAK_GPU_GFX: 'gfx1201', GEAK_GPU_CU_COUNT: '64' }, encoding: 'utf8',
});
ok(counts.includes('GEAK_GPU_CU_COUNT=64') && counts.includes('GEAK_GPU_WGP_COUNT=32'),
   'RDNA4 distinguishes 64 physical CUs from 32 WGP scheduler units');

const fnMatch = workflow.match(/const rdna4BackendAllowed = \(lang, explicitlyRequested\) => \{[\s\S]*?\n\};/);
ok(fnMatch, 'RDNA4 backend policy function exists');
const allowed = Function(`${fnMatch[0]}; return rdna4BackendAllowed;`)();
ok(allowed('hip', false) && allowed('triton', false) && allowed('flydsl', false),
   'HIP, Triton, and direct FlyDSL are default RDNA4 backends');
ok(!allowed('aiter', true) && !allowed('asm', true),
   'AITER and CDNA assembly cannot be enabled on RDNA4');
ok(!allowed('ck', false) && allowed('ck', true), 'CK is explicit opt-in only');
ok(/const tunedSpeedup = IS_RDNA4 \? 0/.test(workflow), 'RDNA4 env-tune winner is hard-disabled');
ok(/export FLYDSL_GPU_ARCH/.test(gpuLock), 'GPU wrapper pins direct FlyDSL architecture');
ok(/gfx120\*\).*rocprofv3 rocprof/.test(profiler), 'RDNA4 profiler order starts with rocprofv3');

for (const fact of ['wave32', 'WGP', 'WMMA', 'GDDR6', 'AITER: no-go', 'FlyDSL main']) {
  ok(rdnaDoc.includes(fact), `RDNA4 hardware card covers ${fact}`);
}
ok(rdnaDoc.includes('perf_knowledge/languages/flydsl/rdna4.md'),
   'RDNA4 hardware card routes FlyDSL work to the architecture authoring card');

const scaledFlydslCap = capabilityIndex.match(
  /- operator: scaled_quant_gemm\n\s+backend: flydsl\n([\s\S]*?)(?=\n  - operator:|$)/);
ok(scaledFlydslCap, 'scaled-quant FlyDSL capability entry exists');
for (const unsafeFact of ['gfx1200', 'gfx1201']) {
  ok(!scaledFlydslCap[1].includes(unsafeFact),
     `scaled-quant machine metadata avoids unsupported cross-product ${unsafeFact}`);
}
ok(!/dtypes: \[[^\n]*\bfp8_e4m3(?:,|\])/.test(scaledFlydslCap[1]),
   'scaled-quant machine metadata avoids unsupported exact dtype fp8_e4m3');
for (const fact of [
  'wave32', '16x16x16', 'v_wmma', '64 KiB', 'FP32 K128', 'broad-prefill',
  '3c03e97919bedbeb95ea803baed089c3725eabb6', '0.9895x',
]) {
  ok(flydslRdna.includes(fact), `FlyDSL RDNA4 card records ${fact}`);
}
ok(scaledFlydsl.includes('Architecture split') && scaledFlydsl.includes('does not establish a broad-M'),
   'scaled-quant card separates RDNA4 capability from performance maturity');
ok(authorRole.includes('languages/flydsl/rdna4.md') && techLeadRole.includes('languages/flydsl/rdna4.md'),
   'planning and author roles explicitly load the FlyDSL RDNA4 card');
ok(expertIndex.includes('id: flydsl_rdna4_fp8_blockscale_small_m') &&
   expertIndex.includes('validation_status: validated'),
   'validated small-M RDNA4 FlyDSL expert recipe is indexed');
ok(rdnaSkill.includes('keeps M1/M2 behind the incumbent') && rdnaSkill.includes('not a broad-M'),
   'expert recipe preserves measured route boundaries and excludes broad prefill');
ok(/const archSafeBackends/.test(e2e) && /rdna4LanguageAllowed/.test(e2e),
   'e2e filters extracted backend candidates through the RDNA4 policy');
ok(/enforceArchBake/.test(e2e) && /b\.tuned_speedup = 0/.test(e2e),
   'e2e cannot promote an AITER\/CK environment winner on RDNA4');

process.stdout.write('RDNA4 support contract: PASS\n');
