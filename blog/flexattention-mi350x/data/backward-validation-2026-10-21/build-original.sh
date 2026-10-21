#!/usr/bin/env bash
set -euo pipefail
cd /home/nmeganat/flex/backward-validation-2026-10-21/pytorch-pr
export PATH="/home/nmeganat/flex/backward-validation-2026-10-21/venv/bin:$PATH"
export USE_ROCM=1 USE_CUDA=0 PYTORCH_ROCM_ARCH=gfx950 BUILD_TEST=0
export USE_DISTRIBUTED=0 USE_FBGEMM=0 USE_FLASH_ATTENTION=0 USE_MEM_EFF_ATTENTION=0
export USE_KINETO=0 USE_ROCM_CK_GEMM=0 USE_ROCM_CK_SDPA=0 USE_MSLK=0
export CMAKE_C_COMPILER_LAUNCHER=/usr/bin/ccache CMAKE_CXX_COMPILER_LAUNCHER=/usr/bin/ccache
export CMAKE_HIP_FLAGS="--rocm-path=/opt/rocm-7.2.1 --hip-path=/opt/rocm-7.2.1 -isystem /opt/rocm-7.2.1/include"
export MAX_JOBS=96 CMAKE_BUILD_PARALLEL_LEVEL=96
python tools/amd_build/build_amd.py
python -m pip install -e . -v --no-build-isolation
