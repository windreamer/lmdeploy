import os
import re
import subprocess
import sys
from pathlib import Path

from packaging.requirements import Requirement

_target_device = os.getenv('LMDEPLOY_TARGET_DEVICE', 'cuda').lower()
_disable_turbomind = os.getenv('DISABLE_TURBOMIND', '').lower()
_true_values = (
    '1',
    'on',
    't',
    'true',
    'yes',
)


def get_turbomind_deps():
    if os.name == 'nt':
        return []

    CUDA_COMPILER = os.getenv('CUDACXX', os.getenv('CMAKE_CUDA_COMPILER', 'nvcc'))
    nvcc_output = subprocess.check_output([CUDA_COMPILER, '--version'], stderr=subprocess.DEVNULL).decode()
    (CUDAVER, ) = re.search(r'release\s+(\d+).', nvcc_output).groups()
    return [
        f'nvidia-nccl-cu{CUDAVER}',
        f'nvidia-cuda-runtime-cu{CUDAVER}',
        f'nvidia-cublas-cu{CUDAVER}',
        f'nvidia-curand-cu{CUDAVER}',
    ]


def parse_requirements(fname):
    reqs = set()
    with open(fname, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                reqs.add(str(Requirement(line)))

    return list(reqs)


_dependencies = parse_requirements(f'requirements/runtime_{_target_device}.txt')

if _target_device == 'cuda' and _disable_turbomind not in _true_values:
    import cmake_build_extension

    ext_modules = [
        cmake_build_extension.CMakeExtension(
            name='_turbomind',
            install_prefix='lmdeploy/lib',
            cmake_depends_on=['pybind11'],
            source_dir=str(Path(__file__).parent.absolute()),
            cmake_generator=None if os.name == 'nt' else 'Ninja',
            cmake_build_type=os.getenv('CMAKE_BUILD_TYPE', 'RelWithDebInfo'),
            cmake_configure_options=[
                f'-DPython3_ROOT_DIR={Path(sys.prefix)}',
                f'-DPYTHON_EXECUTABLE={Path(sys.executable)}',
                '-DCALL_FROM_SETUP_PY:BOOL=ON',
                '-DBUILD_SHARED_LIBS:BOOL=OFF',
                # Select the bindings implementation
                '-DBUILD_PY_FFI=ON',
                '-DBUILD_MULTI_GPU=' + ('OFF' if os.name == 'nt' else 'ON'),
                '-DUSE_NVTX=' + ('OFF' if os.name == 'nt' else 'ON'),
            ],
        ),
    ]
    cmdclass = dict(build_ext=cmake_build_extension.BuildExtension, )
    _dependencies += get_turbomind_deps()
else:
    ext_modules = []
    cmdclass = {}


def pdm_build_initialize(context):
    metadata = context.config.metadata
    metadata['dependencies'] += _dependencies
    extras_require = metadata['optional-dependencies']
    extras_require['all'] = list(set(_dependencies + extras_require.get('serve', []) + extras_require.get('lite', [])))


def pdm_build_update_setup_kwargs(context, setup_kwargs):
    setup_kwargs.update(
        ext_modules=ext_modules,
        cmdclass=cmdclass,
    )
