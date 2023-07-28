import os
import sys
import pkg_resources
import subprocess
import glob
import hashlib
import json
import re
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

slangcPath = os.path.join(
    package_dir, 'bin', 'slangc'+executable_extension)

# If we have SLANGC_PATH set, use that instead
if 'SLANGC_PATH' in os.environ:
    slangcPath = os.environ['SLANGC_PATH']

def _replaceFileExt(fileName, newExt, suffix=None):
    baseName, old_extension = os.path.splitext(fileName)
    if suffix:
        new_filename = baseName + suffix + newExt
    else:
        new_filename = baseName + newExt
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

def get_dictionary_hash(dictionary, truncate_at=16):
    # Convert dictionary to JSON string
    jsonString = json.dumps(dictionary, sort_keys=True)

    # Compute SHA-256 hash of the JSON string
    hashObject = hashlib.sha256(jsonString.encode())
    hashCode = hashObject.hexdigest()

    # Truncate the hash code
    return hashCode[:truncate_at]

def convert_non_alphanumeric_to_underscore(name):
    converted_name = re.sub(r'\W+', '_', name)
    return converted_name

def _computeSlangHash(fileName, targetMode, options, verbose=False):
    compileCommand = [slangcPath, fileName, *options, 
                      '-target', targetMode, '-line-directive-mode', 'none', '-report-hash-only', 'true']
    if verbose:
        print(f"Querying shader hash (target={targetMode}): ", " ".join(compileCommand))

    result = subprocess.run(compileCommand, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    if verbose:
        slangcOutput = result.stderr.decode('utf-8')
        if slangcOutput.strip():
            print(slangcOutput)
        
    if result.returncode != 0:
        raise RuntimeError(f"Query failed with error code {result.returncode}")
    
    hashCode = result.stdout.hex()
    return hashCode

def computeSlangHash(fileName, targetMode, options, verbose=False):
    try:
        return _computeSlangHash(fileName, targetMode, options, verbose)
    except RuntimeError as err:
        if verbose:
            print(f"{err}")
        return None

def makeOptionsList(defines):
    if defines is None:
        return []
    
    defines = dict(defines)
    return list([f"-D{key}={value}" for (key, value) in defines.items()])

def loadModule(fileName, skipSlang=False, verbose=False, defines={}):
    if verbose:
        print(f"Loading slang module: {fileName}")
        print(f"Using slangc.exe location: {slangcPath}")

    if defines:
        optionsHash = "-".join([get_dictionary_hash(defines, truncate_at=16)])
    else:
        optionsHash = None
    
    parentFolder = os.path.dirname(fileName)
    baseNameWoExt = os.path.splitext(os.path.basename(fileName))[0]
    baseOutputFolder = os.path.join(parentFolder, ".slangpy_cache", baseNameWoExt)

    # Specialize output folder with hash of the specialization parameters
    outputFolder = os.path.join(baseOutputFolder, optionsHash)

    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)

    # Try to find a metadata file "metadata.json" in outputFolder.
    metadataFile = os.path.join(outputFolder, "metadata.json")
    metadata = {}
    if os.path.exists(metadataFile):
        metadata = json.load(open(metadataFile, 'r'))

    baseName = os.path.basename(fileName)
    cppOutName = os.path.join(outputFolder, _replaceFileExt(baseName, ".cpp"))
    cudaOutName = os.path.join(outputFolder, _replaceFileExt(baseName, "_cuda.cu"))

    # Check if the defines match the cached metadata.
    definesChanged = not (makeOptionsList(metadata.get("defines", None)) == makeOptionsList(defines))
    if (verbose and os.path.exists(baseOutputFolder)) and definesChanged:
        print("Cache miss (defines)!")

    # Common options
    options = makeOptionsList(defines)
    
    compileStartTime = time.perf_counter()

    # Check if the cpp & cuda target hashes have changed.
    cppTargetHash = computeSlangHash(fileName, "torch-binding", options, verbose)
    if cppTargetHash is not None:    
        cppTargetChanged = not (metadata.get("cppTargetHash", None) == cppTargetHash)
    else:
        if verbose:
            print("Failed to compute hash. Defaulting to cache miss")
        cppTargetChanged = True

    if (verbose and "cppTargetHash" in metadata.keys()) and cppTargetChanged:
        print("Cache miss (host module)!")

    cudaTargetHash = computeSlangHash(fileName, "cuda", options, verbose)
    if cudaTargetHash is not None:    
        cudaTargetChanged = not (metadata.get("cudaTargetHash", None) == cudaTargetHash)
    else:
        if verbose:
            print("Failed to compute hash. Defaulting to cache miss")
        cudaTargetChanged = True

    if (verbose and "cudaTargetHash" in metadata.keys()) and cudaTargetChanged:
        print("Cache miss (kernel module)!")

    if cppTargetChanged or definesChanged:
        compileCommand = [slangcPath, fileName, *options, '-o', cppOutName, 
                          '-target', 'torch-binding', '-line-directive-mode', 'none']
        if verbose:
            print("Building Host Module: ", " ".join(compileCommand))

        result = subprocess.run(compileCommand, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        slangcErr = result.stderr.decode('utf-8')
        if slangcErr.strip():
            print(slangcErr)
        if result.returncode != 0:
            raise RuntimeError(f"Compilation failed with error {result.returncode}")
        
        # Update metadata.
        metadata["defines"] = defines
        metadata["cppTargetHash"] = cppTargetHash
    else:
        if verbose:
            print(f"Using cached host module ({cppTargetHash})")
    
    if cudaTargetChanged or definesChanged:
        compileCommand = [slangcPath, fileName, *options, '-o', cudaOutName, '-line-directive-mode', 'none']
        if verbose:
            print("Building Kernel Module: ", " ".join(compileCommand))

        result = subprocess.run(compileCommand, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        slangcErr = result.stderr.decode('utf-8')
        if slangcErr.strip():
            print(slangcErr)
        if result.returncode != 0:
            raise RuntimeError(f"Compilation failed with error {result.returncode}")
        
        # Update metadata.
        metadata["defines"] = defines
        metadata["cudaTargetHash"] = cudaTargetHash
    else:
        if verbose:
            print(f"Using cached kernel module ({cudaTargetHash})")
    
    # Write metadata file.
    json.dump(metadata, open(metadataFile, 'w'))

    downstreamStartTime = time.perf_counter()
    
    baseModuleName = convert_non_alphanumeric_to_underscore(
        os.path.splitext(os.path.basename(fileName))[0])
    
    # Construct a unique module name by incrementing a persistent counter
    # for each module.
    #
    if loadModule._moduleCounterMap is None:
        loadModule._moduleCounterMap = {}
    
    if baseModuleName not in loadModule._moduleCounterMap:
        loadModule._moduleCounterMap[baseModuleName] = 0
    else:
        loadModule._moduleCounterMap[baseModuleName] += 1
    
    moduleName = f"{baseModuleName}_{loadModule._moduleCounterMap[baseModuleName]}"
    
    # make sure to add cl.exe to PATH on windows so ninja can find it.
    _add_msvc_to_env_var()

    slangLib = load(
        name=moduleName,
        sources=[cppOutName, cudaOutName],
        verbose=verbose,
        build_directory=os.path.realpath(outputFolder))

    downstreamEndTime = time.perf_counter()

    if verbose:
        print(f"Slang compilation time: {downstreamStartTime-compileStartTime:.3f}s")
        print(f'Downstream compile time: {downstreamEndTime-downstreamStartTime:.3f}s')
        
    return slangLib

# Initialize module counter map.
loadModule._moduleCounterMap = {}