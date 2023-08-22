import os
import sys
import pkg_resources
import subprocess
import glob
import hashlib
import json
import re
import time

from .util import jit_compile

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

MODULE_VERSIONS = {}

def getUniqueSessionVersion(moduleKey):
    if moduleKey not in MODULE_VERSIONS:
        MODULE_VERSIONS[moduleKey] = 0
    else:
        MODULE_VERSIONS[moduleKey] += 1
    
    return MODULE_VERSIONS[moduleKey]

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

def makeOptionsList(defines):
    if defines is None:
        return []
    
    defines = dict(defines)
    return list([f"-D{key}={value}" for (key, value) in defines.items()])

def makeBuildDirPath(baseDir, buildID):
    return os.path.join(baseDir, f"{buildID}")

def getLatestDir(moduleKey, baseDir):
    latestFile = os.path.join(baseDir, "latest.txt")
    if os.path.exists(latestFile):
        with open(latestFile, 'r') as f:
            latestBuildID = int(f.read())
    else:
        latestBuildID = 0
        with open(latestFile, 'w') as f:
            f.write("0")
    
    return makeBuildDirPath(baseDir, latestBuildID)

def getOrCreateUniqueDir(moduleKey, baseDir):
    # Check if buildDir has a latest.txt file. If so, read the contents.
    # If not, create a latest.txt with '0' as the contents.
    #
    latestFile = os.path.join(baseDir, "latest.txt")
    if os.path.exists(latestFile):
        with open(latestFile, 'r') as f:
            latestBuildID = int(f.read())
    else:
        latestBuildID = 0
        with open(latestFile, 'w') as f:
            f.write("0")

    targetBuildID = getUniqueSessionVersion(moduleKey)
    
    if targetBuildID == latestBuildID:
        # If targetBuildID is the same as latestBuildID, we can reuse the same clone.
        dirname = makeBuildDirPath(baseDir, targetBuildID)

        # The directory should already exist, but if not, just make it.
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        return dirname
    else:
        # Copy all contents of the latestBuildID folder to the targetBuildID folder.
        # including metadata!
        #
        latestDir = makeBuildDirPath(baseDir, latestBuildID)
        targetDir = makeBuildDirPath(baseDir, targetBuildID)

        if os.path.exists(targetDir):
            import shutil
            shutil.rmtree(targetDir)

        import distutils.dir_util
        distutils.dir_util.copy_tree(latestDir, targetDir)

        # Update latest.txt
        with open(latestFile, 'w') as f:
            f.write(str(targetBuildID))
        
        return targetDir

def compileSlang(metadata, fileName, targetMode, options, outputFile, verbose=False, dryRun=False):
    needsRecompile = False

    # If any of the depfiles are newer than outputFile, we need to recompile.
    if metadata and metadata.get("deps"):
        depFiles = metadata["deps"]
        if not os.path.exists(outputFile):
            if verbose:
                print(f"Output file {outputFile} does not exist. Needs recompile.", file=sys.stderr)
            needsRecompile = True
        else:
            for depFile, timestamp in depFiles:
                if verbose:
                    print(f"Checking dependency: {depFile}", file=sys.stderr)
                
                if not os.path.exists(depFile):
                    if verbose:
                        print(f"\tDependency does not exist. Needs recompile.", file=sys.stderr)
                    needsRecompile = True
                    break
                
                if os.path.getmtime(depFile) > timestamp:
                    if verbose:
                        print(f"\tDependency is newer. Needs recompile.", file=sys.stderr)
                    needsRecompile = True
                    break
    else:
        needsRecompile = True

    # If any of the options are different, we need to recompile.
    if metadata and (metadata.get("options", None) != None):
        oldOptions = metadata["options"]
        if verbose:
            print("Checking options... ", file=sys.stderr)
        if oldOptions != options:
            if verbose:
                print(f"Options are different \"{oldOptions}\" => \"{options}\". Needs recompile.", file=sys.stderr)
            needsRecompile = True
    else:
        needsRecompile = True
    
    if needsRecompile:
        return True, (_compileSlang(metadata, fileName, targetMode, options, outputFile, verbose) if not dryRun else None)
    else:
        return False, (metadata if not dryRun else None)

def _compileSlang(metadata, fileName, targetMode, options, outputFile, verbose=False):
    # Create a temporary depfile path.
    depFile = f"{outputFile}.d.out"

    compileCommand = [slangcPath, fileName, *options, 
                      '-target', targetMode,
                      '-line-directive-mode', 'none',
                      '-o', outputFile,
                      '-depfile', depFile]
    if verbose:
        print(f"Building {os.path.basename(fileName)} -> {os.path.basename(outputFile)}: ", 
              " ".join(compileCommand), file=sys.stderr)

    result = subprocess.run(compileCommand, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    slangcErr = result.stderr.decode('utf-8')
    if slangcErr.strip():
        print(slangcErr, file=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"Compilation failed with error {result.returncode}")
    
    deps = parseDepfile(depFile)

    # Erase depfile.
    os.remove(depFile)

    # Update metadata.
    return { "options": options, "deps": deps}

def compileAndLoadModule(metadata, sources, moduleName, buildDir, verbose=False, dryRun=False):
    needsRebuild = False

    newMetadata = metadata.copy()

    # Check if any of the sources are newer than the module binary.
    if metadata and metadata.get("moduleBinary", None):
        moduleBinary = os.path.realpath(metadata["moduleBinary"])
        if os.path.exists(moduleBinary):
            for source in sources:
                if verbose:
                    print("Checking source: ", source, file=sys.stderr)
                    
                if not os.path.exists(source):
                    raise RuntimeError(f"Source file {source} does not exist")

                if os.path.getmtime(source) > os.path.getmtime(moduleBinary):
                    if verbose:
                        print("Source file is newer than module binary. Rebuilding.", file=sys.stderr)
                    needsRebuild = True
                    break
        else:
            needsRebuild = True
    else:
        needsRebuild = True

    cacheLookupKey = moduleName

    if not needsRebuild:
        # Try the session cache. If we find a hit, the module is already loaded. 
        if compileAndLoadModule._moduleCache is not None:
            if cacheLookupKey in compileAndLoadModule._moduleCache:
                if verbose:
                    print(f"Skipping build. Using cached module ({cacheLookupKey})", file=sys.stderr)
                if dryRun:
                    return False, None
                return compileAndLoadModule._moduleCache[cacheLookupKey], newMetadata
        
        # If not, try the persistent cache. It's a lot quicker to import the binary
        # than going through torch's build process again (even if the build detects 
        # that there is nothing to do)
        #
        moduleBinary = os.path.realpath(metadata["moduleBinary"])
        if os.path.exists(moduleBinary):
            if dryRun:
                return False, None
            
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    metadata["moduleName"], moduleBinary)
                slangLib = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(slangLib)
            except Exception as e:
                if verbose:
                    print(f"Failed to load existing module binary {moduleBinary}: {e}", file=sys.stderr)
                    print(f"Needs rebuild", file=sys.stderr)

                needsRebuild = True
        else:
            needsRebuild = True

    if needsRebuild:
        if dryRun:
            return True, None
        
        # Compile the module.
        slangLib = _compileAndLoadModule(metadata, sources, moduleName, buildDir, verbose)

        newMetadata = metadata.copy()
        newMetadata["moduleName"] = moduleName
        newMetadata["moduleBinary"] = os.path.join(buildDir, f"{moduleName}.pyd")
    
    if dryRun:
        return False, None
    
    compileAndLoadModule._moduleCache[cacheLookupKey] = slangLib

    return slangLib, newMetadata

compileAndLoadModule._moduleCache = {}

def _compileAndLoadModule(metadata, sources, moduleName, buildDir, verbose=False):
    # make sure to add cl.exe to PATH on windows so ninja can find it.
    _add_msvc_to_env_var()

    return jit_compile(
        moduleName,
        sources,
        extra_cflags=None,
        extra_cuda_cflags=None,
        extra_ldflags=None,
        extra_include_paths=None,
        build_directory=os.path.realpath(buildDir),
        verbose=verbose,
        is_python_module=True,
        is_standalone=False,
        keep_intermediates=True,
        with_cuda=None)

def parseDepfile(depFile):
    with open(depFile, 'r') as f:
        depFileContents = f.readlines()

    allDepFiles = []
    for targetEntry in depFileContents:
        _ = targetEntry.split(": ")[0]
        depFiles = [os.path.realpath(token.replace(r"\:", ":")) for token in targetEntry.split(": ")[1].strip().split(" ")]

        # Convert all depfiles to real paths.
        depFilesWithTimestamps = [
            (depFile, os.path.getmtime(depFile)) for depFile in depFiles]
        
        allDepFiles.extend(depFilesWithTimestamps)

    return allDepFiles

def _loadModule(fileName, moduleName, outputFolder, options, sourceDir=None, verbose=False, dryRun=False):

    # Try to find a metadata file "metadata.json" in outputFolder.
    metadataFile = os.path.join(outputFolder, "metadata.json")
    metadata = {}
    if os.path.exists(metadataFile):
        with open(metadataFile, 'r') as f:
            metadata = json.load(f)

    baseName = os.path.basename(fileName)
    if sourceDir is None:
        cppOutName = os.path.join(outputFolder, _replaceFileExt(baseName, ".cpp"))
        cudaOutName = os.path.join(outputFolder, _replaceFileExt(baseName, "_cuda.cu"))
    else:
        cppOutName = os.path.join(sourceDir, _replaceFileExt(baseName, ".cpp"))
        cudaOutName = os.path.join(sourceDir, _replaceFileExt(baseName, "_cuda.cu"))

    # Compile slang files to intermediate host and kernel modules.
    compileStartTime = time.perf_counter()

    resultCpp, metadataCpp = compileSlang(metadata.get("cpp", None), fileName, "torch-binding", options, cppOutName, verbose, dryRun=dryRun)
    metadata["cpp"] = metadataCpp

    resultCuda, metadataCuda = compileSlang(metadata.get("cuda", None), fileName, "cuda", options, cudaOutName, verbose, dryRun=dryRun)
    metadata["cuda"] = metadataCuda

    if dryRun and (resultCuda or resultCpp):
        return True

    compileEndTime = time.perf_counter()

    # Compile host and kernel modules to torch module.
    downstreamStartTime = time.perf_counter()
    
    slangLib, metadata = compileAndLoadModule(
        metadata, [cppOutName, cudaOutName], 
        moduleName, outputFolder, verbose, dryRun=dryRun)

    if dryRun:
        if slangLib:
            return True
        else:
            return False
    
    downstreamEndTime = time.perf_counter()

    # Save metadata.
    with open(metadataFile, 'w') as f:
        json.dump(metadata, f)

    if verbose:
        print(f"Slang compile time: {compileEndTime-compileStartTime:.3f}s", file=sys.stderr)
        print(f'Downstream compile time: {downstreamEndTime-downstreamStartTime:.3f}s', file=sys.stderr)
    
    return slangLib
            
def loadModule(fileName, skipSlang=None, verbose=False, defines={}):
    # Print warning
    if skipSlang is not None:
        print("Warning: skipSlang is deprecated in favor of a dependency-based cache.", file=sys.stderr)

    if verbose:
        print(f"Loading slang module: {fileName}", file=sys.stderr)
        print(f"Using slangc.exe location: {slangcPath}", file=sys.stderr)

    if defines:
        optionsHash = get_dictionary_hash(defines, truncate_at=16)
    else:
        optionsHash = get_dictionary_hash({}, truncate_at=16)
    
    parentFolder = os.path.dirname(fileName)
    baseNameWoExt = os.path.splitext(os.path.basename(fileName))[0]
    baseOutputFolder = os.path.join(parentFolder, ".slangpy_cache", baseNameWoExt)

    # Specialize output folder with hash of the specialization parameters
    outputFolder = os.path.join(baseOutputFolder, optionsHash)

    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)
    
    # Common options
    options = makeOptionsList(defines)

    # Module name
    moduleName = f"_slangpy_{convert_non_alphanumeric_to_underscore(baseNameWoExt)}_{optionsHash}"
    
    # Dry run with latest build dir
    buildDir = getLatestDir(outputFolder, outputFolder)

    if verbose:
        print(f"Dry-run using latest build directory: {buildDir}", file=sys.stderr)

    needsRecompile = _loadModule(fileName, moduleName, buildDir, options, sourceDir=outputFolder, verbose=verbose, dryRun=True)

    if needsRecompile:
        if verbose:
            print("Rebuild required. Creating unique build directory", file=sys.stderr)
        # Handle versioning
        buildDir = getOrCreateUniqueDir(outputFolder, outputFolder)
    else:
        buildDir = buildDir
    
    if verbose:
        print(f"Working folder: {buildDir}", file=sys.stderr)

    return _loadModule(fileName, moduleName, buildDir, options, sourceDir=outputFolder, verbose=verbose, dryRun=False)

def clearSessionShaderCache():
    compileAndLoadModule._moduleCache = {}

def clearPersistentShaderCache():
    baseOutputFolder = os.path.join(package_dir, '.slangpy_cache')
    if os.path.exists(baseOutputFolder):
        import shutil
        shutil.rmtree(baseOutputFolder)

def clearShaderCaches():
    clearSessionShaderCache()
    clearPersistentShaderCache()