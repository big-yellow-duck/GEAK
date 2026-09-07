# Prefill variant provenance

`round4_kernel.py` is a byte-for-byte copy of the completed GEAK round-4 candidate at:

`exp/bakeoff_baseline_20260824_093634/bakeoff/triton/team_task_20260824_094947_3467089_13985/task/round_4/engineer_0/workspace/kernel_src/kernel.py`

SHA-256: `036a1a467d7aaa3d7920c91696703b8cc2523032edca436442e78a63a418de3b`

The selection benchmark imports its BM32, BM64 fused-scale, and BM80 shared-B kernels, then launches
each symbol directly. The hardcoded router in this provenance copy is never used by the benchmark.
This keeps kernel selection evidence separate from the implementation being measured.
