[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$repoWslOutput = & wsl.exe -e wslpath -a -- $repoRoot
$pathExit = $LASTEXITCODE
$repoWsl = ($repoWslOutput | Select-Object -First 1).Trim()
if ($pathExit -ne 0 -or [string]::IsNullOrWhiteSpace($repoWsl)) {
    throw 'Could not resolve the RouteB checkout path inside WSL.'
}

$pythonCode = "import sys, jax, jaxlib, jax.numpy as jnp; x=jnp.arange(16, dtype=jnp.float32); y=jax.jit(lambda a:jnp.sum(a*a))(x); y.block_until_ready(); devices=jax.devices(); print('python='+sys.version.split()[0]); print('jax='+jax.__version__); print('jaxlib='+jaxlib.__version__); print('devices='+str(devices)); print('jit_sum_squares='+str(float(y)))"
$windowsWorker = Join-Path $repoRoot 'packages\world_model\environment\.venv-win-cpu\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $windowsWorker -PathType Leaf)) {
    throw "RouteB's Windows CPU JAX worker is missing: $windowsWorker. See docs/world_model/V3_IMPLEMENTATION_STATUS.md for the locked installation command."
}

$priorPlatforms = $env:JAX_PLATFORMS
try {
    $env:JAX_PLATFORMS = 'cpu'
    & $windowsWorker -c $pythonCode
    $windowsExit = $LASTEXITCODE
}
finally {
    if ($null -eq $priorPlatforms) {
        Remove-Item Env:JAX_PLATFORMS -ErrorAction SilentlyContinue
    }
    else {
        $env:JAX_PLATFORMS = $priorPlatforms
    }
}
if ($windowsExit -ne 0) {
    throw "RouteB's isolated Windows CPU JAX check failed (exit $windowsExit)."
}

$repoWslQuoted = $repoWsl.Replace("'", "'\''")
$wslWorker = 'packages/world_model/environment/.venv-wsl/bin/python'
$probe = "export XLA_PYTHON_CLIENT_PREALLOCATE=false && cd '$repoWslQuoted' && $wslWorker -c `"$pythonCode`""
& wsl.exe -e bash -lc $probe
if ($LASTEXITCODE -ne 0) {
    throw "RouteB's isolated WSL CUDA JAX check failed (exit $LASTEXITCODE). See docs/world_model/V3_IMPLEMENTATION_STATUS.md for the worker environment."
}
