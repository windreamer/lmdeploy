#!/usr/bin/env bash
# Python builder
# Adopted from https://github.com/dusty-nv/jetson-containers/blob/master/packages/pytorch/build.sh
#
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

set -ex

echo "Building PyTorch ${PYTORCH_BUILD_VERSION}"

# build from source
git clone --branch "v${PYTORCH_BUILD_VERSION}" --depth=1 --recursive https://github.com/pytorch/pytorch /opt/pytorch ||
git clone --depth=1 --recursive https://github.com/pytorch/pytorch /opt/pytorch
cd /opt/pytorch

# https://github.com/pytorch/pytorch/issues/138333
CPUINFO_PATCH=third_party/cpuinfo/src/arm/linux/aarch64-isa.c
sed -i 's|cpuinfo_log_error|cpuinfo_log_warning|' ${CPUINFO_PATCH}
grep 'PR_SVE_GET_VL' ${CPUINFO_PATCH} || echo "patched ${CPUINFO_PATCH}"
tail -20 ${CPUINFO_PATCH}

pip3 install -r requirements.txt
pip3 install scikit-build ninja
pip3 install 'cmake<4'

#TORCH_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=0" \
# https://github.com/pytorch/pytorch/pull/157791/files#diff-f271c3ed0c135590409465f4ad55c570c418d2c0509bbf1b1352ebdd1e6611d1
export TORCH_NVCC_FLAGS="-Xfatbin -compress-all"

PYTORCH_BUILD_NUMBER=1 \
USE_CUDNN=1 \
USE_CUSPARSELT=1 \
USE_CUDSS=1 \
USE_NATIVE_ARCH=1 \
USE_FLASH_ATTENTION=1 \
USE_MEM_EFF_ATTENTION=1 \
USE_TENSORRT=0 \
USE_BLAS="$USE_BLAS" \
BLAS="$BLAS" \
python3 setup.py bdist_wheel --dist-dir /opt

cd /
rm -rf /opt/pytorch

# install the compiled wheel
pip3 install /opt/torch*.whl
python3 -c 'import torch; print(f"PyTorch version: {torch.__version__}"); print(f"CUDA available:  {torch.cuda.is_available()}"); print(f"cuDNN version:   {torch.backends.cudnn.version()}"); print(torch.__config__.show());'
twine upload --verbose /opt/torch*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
