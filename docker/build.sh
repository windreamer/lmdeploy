#!/bin/bash -ex

apt-get update -y
apt-get install -y --no-install-recommends \
    software-properties-common wget curl openssh-server ssh sudo \
    git-core libibverbs1 ibverbs-providers ibverbs-utils librdmacm1 libibverbs-dev rdma-core libmlx5-1
add-apt-repository -y ppa:deadsnakes/ppa
apt-get install -y --no-install-recommends \
    rapidjson-dev libgoogle-glog-dev gdb python${PYTHON_VERSION}-minimal python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv
apt-get clean -y
rm -rf /var/lib/apt/lists/*

pushd /opt >/dev/null
    python${PYTHON_VERSION} -m venv py3
popd >/dev/null

export PATH=/opt/py3/bin:$PATH

if [[ "${CUDA_VERSION_SHORT}" = "cu128" ]]; then
    NVCC_GENCODE="-gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_90,code=sm_90 -gencode=arch=compute_120,code=sm_120 -gencode=arch=compute_120,code=compute_120"
else
    NVCC_GENCODE="-gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_90,code=sm_90 -gencode=arch=compute_90,code=compute_90"
fi

pushd /tmp >/dev/null
    git clone --depth=1 --branch ${NCCL_BRANCH} https://github.com/NVIDIA/nccl.git
    pushd nccl >/dev/null
        make NVCC_GENCODE="$NVCC_GENCODE" -j$(nproc) src.build
        mv build/include/* /usr/local/include
        mkdir -p /usr/local/nccl/lib
        mv build/lib/lib* /usr/local/nccl/lib/
    popd >/dev/null
popd >/dev/null
rm -rf /tmp/nccl

export LD_LIBRARY_PATH=/usr/local/nccl/lib:$LD_LIBRARY_PATH

pip install --upgrade pip build
python3 -m build -w -o /wheels -v .

if [[ "${CUDA_VERSION_SHORT}" != "cu118" ]]; then
    GDRCOPY_VERSION=2.4.4
    NVSHMEM_VERSION=3.2.5-1
    DEEP_EP_VERSION=bdd119f
    FLASH_MLA_VERSION=9edee0c
    DEEP_GEMM_VERSION=1876566

    pushd /tmp >/dev/null
        curl -sSL 'https://github.com/NVIDIA/gdrcopy/archive/refs/tags/v${GDRCOPY_VERSION}.tar.gz' | tar xz
        pushd gdrcopy-${GDRCOPY_VERSION} >/dev/null
            make prefix=/usr/local/gdrcopy -j $(proc) install
        popd >/dev/null
        rm -rf gdrcopy-${GDRCOPY_VERSION}

        pip install torch cmake

        git clone --depth=1 https://github.com/deepseek-ai/DeepEP.git
        pushd DeepEP >/dev/null
            git checkout ${NVSHMEM_VERSION}
        popd >/dev/null


        # NVSHMEM
        curl -sSL 'https://developer.nvidia.com/downloads/assets/secure/nvshmem/nvshmem_src_${NVSHMEM_VERSION}.txz' | tar xJz
        pushd nvshmem_src >/dev/null
            git apply /tmp/DeepEP/third-party/nvshmem.patch
            NVSHMEM_SHMEM_SUPPORT=0 \
            NVSHMEM_UCX_SUPPORT=0 \
            NVSHMEM_USE_NCCL=0 \
            NVSHMEM_MPI_SUPPORT=0 \
            NVSHMEM_IBGDA_SUPPORT=1 \
            NVSHMEM_PMIX_SUPPORT=0 \
            NVSHMEM_TIMEOUT_DEVICE_POLLING=0 \
            NVSHMEM_USE_GDRCOPY=1 \
            cmake -S . -B build/ -DCMAKE_INSTALL_PREFIX=/usr/local/nvshmem -DMLX5_lib=/lib/x86_64-linux-gnu/libmlx5.so.1
            cmake --build build --target install --parallel $(nproc)
        popd >/dev/null
        rm -rf nvshmem_src

        NVSHMEM_DIR=/usr/local/nvshmem pip wheel -v -w /wheels DeepEP
        rm -rf DeepEP

        pip wheel -v -w /wheels git+https://github.com/deepseek-ai/FlashMLA.git@${FLASH_MLA_VERSION}
        pip wheel -v -w /wheels git+https://github.com/deepseek-ai/DeepGEMM.git@${DEEP_GEMM_VERSION}

    popd >/dev/null
else
    mkdir -p /usr/local/gdrcopy
    mkdir -p /usr/local/nvshmem
fi
