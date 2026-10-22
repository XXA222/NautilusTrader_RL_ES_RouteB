[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$python = Join-Path $repoRoot 'packages\world_model\environment\.venv-win-cpu\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "RouteB's isolated Windows JAX worker is missing: $python. See docs/world_model/V3_IMPLEMENTATION_STATUS.md for the pinned environment setup."
}

$oldPythonPath = $env:PYTHONPATH
$oldJaxPlatforms = $env:JAX_PLATFORMS
$oldPreallocate = $env:XLA_PYTHON_CLIENT_PREALLOCATE
try {
    $env:PYTHONPATH = @(
        (Join-Path $repoRoot 'packages\world_model\src'),
        (Join-Path $repoRoot 'packages\data\src'),
        (Join-Path $repoRoot 'packages\contracts\src'),
        (Join-Path $repoRoot 'packages\orchestrator\src'),
        (Join-Path $repoRoot 'packages\research\src')
    ) -join ';'
    $env:JAX_PLATFORMS = 'cpu'
    $env:XLA_PYTHON_CLIENT_PREALLOCATE = 'false'

    Push-Location $repoRoot
    try {
        Write-Host "Running JAX tests with RouteB worker: $python"
        & $python -m pytest -q `
            tests/world_model/test_routeb_additive.py::test_checkpoint_roundtrip_full_optimizer_tree `
            tests/world_model/test_routeb_e1.py::test_e1_train_loss_is_finite_and_respects_target_mask `
            tests/world_model/test_routeb_hierarchy_weights.py::test_hierarchical_kl_weights_apply_once_per_component `
            tests/world_model/test_routeb_hierarchy_weights.py::test_risk_horizon_weights_emphasize_long_horizon_losses `
            tests/world_model/test_routeb_hierarchy_train.py::test_hierarchical_training_rejects_future_hour_state `
            tests/world_model/test_routeb_hierarchy_train.py::test_hierarchical_training_rejects_future_day_condition_for_hour `
            tests/world_model/test_routeb_hierarchy_train.py::test_hierarchical_window_keeps_slow_states_causal_and_fixed_shape `
            tests/world_model/test_routeb_quality_v2.py::test_checkpoint_detects_corruption_and_tree_permutation `
            tests/world_model/test_routeb_quality_v2.py::test_risk_conditioned_return_scale_matches_native_log_rv_in_raw_units `
            tests/world_model/test_routeb_r2_regularizer.py
        if ($LASTEXITCODE -ne 0) {
            throw "RouteB JAX-specific world-model tests failed (exit $LASTEXITCODE)."
        }
        & $python qa/world_model/routeb_r2_regularizer_smoke.py --checkout .tmp/dreamerv3-route-b
        if ($LASTEXITCODE -ne 0) {
            throw "RouteB R2 regularizer integration smoke failed (exit $LASTEXITCODE)."
        }
    }
    finally {
        Pop-Location
    }
}
finally {
    if ($null -eq $oldPythonPath) { Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue } else { $env:PYTHONPATH = $oldPythonPath }
    if ($null -eq $oldJaxPlatforms) { Remove-Item Env:JAX_PLATFORMS -ErrorAction SilentlyContinue } else { $env:JAX_PLATFORMS = $oldJaxPlatforms }
    if ($null -eq $oldPreallocate) { Remove-Item Env:XLA_PYTHON_CLIENT_PREALLOCATE -ErrorAction SilentlyContinue } else { $env:XLA_PYTHON_CLIENT_PREALLOCATE = $oldPreallocate }
}
