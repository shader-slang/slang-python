import os
import sys
import pkg_resources
import subprocess
import glob
import time

from torch.utils.cpp_extension import load

package_dir = pkg_resources.resource_filename(__name__, '')

if sys.platform == "win32":
    # Windows
    executable_extension = ".exe"

elif sys.platform == "darwin":
    # macOS
    executable_extension = ""
else:
    # Linux and other Unix-like systems
    executable_extension = ""

slangc_path = os.path.join(
    package_dir, 'bin', 'slangc'+executable_extension)

def _replaceFileExt(fileName, newExt):
    base_name, old_extension = os.path.splitext(fileName)
    new_filename = base_name + newExt
    return new_filename

def find_cl():
    # Look for cl.exe in the default installation path for Visual Studio
    vswhere_path = os.environ.get('ProgramFiles(x86)', '') + '\\Microsoft Visual Studio\\Installer\\vswhere.exe'

    # Get the installation path of the latest version of Visual Studio
    result = subprocess.run([vswhere_path, '-latest', '-property', 'installationPath'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    vs_install_path = result.stdout.decode('utf-8').rstrip()

    # Find the path to cl.exe
    cl_path = glob.glob(os.path.join(vs_install_path, "**", "VC", "Tools", "MSVC", "**", "bin", "HostX64", "X64"), recursive=True)

    if not cl_path:
        raise ValueError("cl.exe not found in default Visual Studio installation path")

    # Get the latest version of cl.exe
    cl_path.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return cl_path[0]

def _add_msvc_to_env_var():
    if sys.platform == 'win32':
        path_to_add = find_cl()
        if path_to_add not in os.environ["PATH"].split(os.pathsep):
            os.environ["PATH"] += os.pathsep + path_to_add

def loadModule(fileName, skipSlang=False, verbose=False):
    if verbose:
        print("loading slang module: " + fileName)
        print("slangc location: " + slangc_path)

    parent_folder = os.path.dirname(fileName)
    output_folder = os.path.join(parent_folder, ".slangpy_cache")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    base_name = os.path.basename(fileName)
    cppOutName = os.path.join(output_folder, _replaceFileExt(base_name, ".cpp"))
    cudaOutName = os.path.join(output_folder, _replaceFileExt(base_name, "_cuda.cu"))

    compileStartTime = time.perf_counter()

    if not(skipSlang and os.path.exists(cppOutName)):
        result = subprocess.run([slangc_path, fileName, '-o', cppOutName, '-target', 'torch-binding' ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        slangcOutput = result.stderr.decode('utf-8')
        if slangcOutput.strip():
            print(slangcOutput)
        if result.returncode != 0:
            raise RuntimeError(f"compilation failed with error {result.returncode}")
    
    if not(skipSlang and os.path.exists(cudaOutName)):
        result = subprocess.run([slangc_path, fileName, '-o', cudaOutName ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        slangcOutput = result.stderr.decode('utf-8')
        if slangcOutput.strip():
            print(slangcOutput)
        if result.returncode != 0:
            raise RuntimeError(f"compilation failed with error {result.returncode}")
    
    downstreamStartTime = time.perf_counter()

    moduleName = os.path.basename(fileName)[0]
    
    # make sure to add cl.exe to PATH on windows so ninja can find it.
    _add_msvc_to_env_var()

    slangLib = load(name=moduleName, sources=[cppOutName,cudaOutName])

    downstreamEndTime = time.perf_counter()

    if verbose:
        print(f"Slang compilation time: {downstreamStartTime-compileStartTime}s")
        print(f'Downstream compile time: {downstreamEndTime-downstreamStartTime}')
        
    return slangLib
