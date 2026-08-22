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
ok(/const archSafeBackends/.test(e2e) && /rdna4LanguageAllowed/.test(e2e),
   'e2e filters extracted backend candidates through the RDNA4 policy');
ok(/enforceArchBake/.test(e2e) && /b\.tuned_speedup = 0/.test(e2e),
   'e2e cannot promote an AITER\/CK environment winner on RDNA4');

process.stdout.write('RDNA4 support contract: PASS\n');
