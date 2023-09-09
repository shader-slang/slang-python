#
# Modified version of _jit_compile from torch.utils.cpp_extension that does not 
# implement versioning.
#

from torch.utils.cpp_extension import (
    _write_ninja_file_and_build_library,
    _import_module_from_library,
    _get_exec_path,
    _join_rocm_home,
    _is_cuda_file,
    _TORCH_PATH,
    JIT_EXTENSION_VERSIONER,
    IS_HIP_EXTENSION
)

from torch.utils.file_baton import FileBaton
from torch.utils.hipify import hipify_python

from typing import Optional
import sys
import os

def jit_compile(name,
                 sources,
                 extra_cflags,
                 extra_cuda_cflags,
                 extra_ldflags,
                 extra_include_paths,
                 build_directory: str,
                 verbose: bool,
                 with_cuda: Optional[bool],
                 is_python_module,
                 is_standalone,
                 keep_intermediates=True) -> None:
    if is_python_module and is_standalone:
        raise ValueError("`is_python_module` and `is_standalone` are mutually exclusive.")

    if with_cuda is None:
        with_cuda = any(map(_is_cuda_file, sources))
    with_cudnn = any(['cudnn' in f for f in extra_ldflags or []])

    baton = FileBaton(os.path.join(build_directory, 'lock'))
    if baton.try_acquire():
        try:
            with hipify_python.GeneratedFileCleaner(keep_intermediates=keep_intermediates) as clean_ctx:
                if IS_HIP_EXTENSION and (with_cuda or with_cudnn):
                    hipify_result = hipify_python.hipify(
                        project_directory=build_directory,
                        output_directory=build_directory,
                        header_include_dirs=(extra_include_paths if extra_include_paths is not None else []),
                        extra_files=[os.path.abspath(s) for s in sources],
                        ignores=[_join_rocm_home('*'), os.path.join(_TORCH_PATH, '*')],  # no need to hipify ROCm or PyTorch headers
                        show_detailed=verbose,
                        show_progress=verbose,
                        is_pytorch_extension=True,
                        clean_ctx=clean_ctx
                    )

                    hipified_sources = set()
                    for source in sources:
                        s_abs = os.path.abspath(source)
                        hipified_sources.add(hipify_result[s_abs]["hipified_path"] if s_abs in hipify_result else s_abs)

                    sources = list(hipified_sources)

                _write_ninja_file_and_build_library(
                    name=name,
                    sources=sources,
                    extra_cflags=extra_cflags or [],
                    extra_cuda_cflags=extra_cuda_cflags or [],
                    extra_ldflags=extra_ldflags or [],
                    extra_include_paths=extra_include_paths or [],
                    build_directory=build_directory,
                    verbose=verbose,
                    with_cuda=with_cuda,
                    is_standalone=is_standalone)
        finally:
            baton.release()
    else:
        baton.wait()

    if verbose:
        print(f'Loading extension module {name}...', file=sys.stderr)

    if is_standalone:
        return _get_exec_path(name, build_directory)

    return _import_module_from_library(name, build_directory, is_python_module)