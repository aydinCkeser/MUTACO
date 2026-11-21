[CmdletBinding(PositionalBinding = $false)]
param(
    [ValidateSet('mutaco-sbi', 'mutaco-dev')]
    [string]$Environment = 'mutaco-sbi',

    [switch]$AllowNonSbi,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$CommandArgs
)

$ErrorActionPreference = 'Stop'
$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path

if ($Environment -eq 'mutaco-dev') {
    if (-not $AllowNonSbi) {
        throw 'mutaco-dev requires -AllowNonSbi; Paper, SBI, TRUBA, and pytest workflows require mutaco-sbi'
    }
    $joined = $CommandArgs -join ' '
    if ($joined -match 'pytest|experiments\.paper|experiments\.graph_inversion|infrastructure\.truba|\bsbi\b') {
        throw 'mutaco-dev cannot run pytest, Paper/SBI, graph-inversion, or TRUBA workflows'
    }
}

$overrideName = if ($Environment -eq 'mutaco-sbi') { 'MUTACO_SBI_PYTHON' } else { 'MUTACO_DEV_PYTHON' }
$python = [Environment]::GetEnvironmentVariable($overrideName)
if (-not $python) {
    $conda = Get-Command conda -ErrorAction Stop
    $condaBase = (& $conda.Source info --base).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $condaBase) {
        throw 'unable to resolve the Conda installation root'
    }
    $python = Join-Path $condaBase "envs\$Environment\python.exe"
}
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "$Environment interpreter is missing: $python"
}
if (-not $CommandArgs -or $CommandArgs.Count -eq 0) {
    throw 'at least one Python argument is required after the launcher boundary'
}

# Immutable declarations and the checkout import address require real Git links.
foreach ($relative in @('experiments\analytic_target_matching\src\mutaco_analytic\__init__.py', 'experiments\graph_inversion\frozen_targets\paper120_v3\manifest.json')) {
    if (-not (Test-Path -LiteralPath (Join-Path $repositoryRoot $relative) -PathType Leaf)) {
        throw "Required checkout path is unavailable: $relative. On Windows, enable symlink creation and use a checkout created with git -c core.symlinks=true clone."
    }
}

$sourceRoots = @(
    (Join-Path $repositoryRoot 'src'),
    (Join-Path $repositoryRoot 'experiments\analytic_target_matching\src'),
    (Join-Path $repositoryRoot 'simulators\graph-simulator\src')
)
$prefix = $sourceRoots -join [IO.Path]::PathSeparator
$env:PYTHONPATH = if ($env:PYTHONPATH) {
    "$prefix$([IO.Path]::PathSeparator)$env:PYTHONPATH"
} else {
    $prefix
}
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'

$probe = @'
import json
from pathlib import Path
import sys

expected = sys.argv[1]
repository = Path(sys.argv[2]).resolve()
if Path(sys.prefix).name.lower() != expected:
    raise SystemExit(f"refusing interpreter prefix {sys.prefix!r}; expected environment {expected!r}")
import torch
import mutaco
if expected == "mutaco-sbi":
    import sbi
else:
    sbi = None
module_path = Path(mutaco.__file__).resolve()
if repository not in module_path.parents:
    raise SystemExit(f"mutaco imported outside this checkout: {module_path}")
print(json.dumps({
    "environment": expected,
    "python": sys.executable,
    "torch": torch.__version__,
    "sbi": None if sbi is None else sbi.__version__,
    "mutaco": str(module_path),
}, sort_keys=True))
'@

$probeErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
$probeOutput = $probe | & $python - $Environment $repositoryRoot
$probeExitCode = $LASTEXITCODE
$ErrorActionPreference = $probeErrorActionPreference
if ($probeExitCode -ne 0) {
    throw "$Environment dependency validation failed"
}
Write-Host "MUTACO_RUNTIME=$probeOutput"

Push-Location $repositoryRoot
try {
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    $previousLaunchArguments = [Environment]::GetEnvironmentVariable('MUTACO_LAUNCH_ARGUMENTS_JSON')
    $env:MUTACO_LAUNCH_ARGUMENTS_JSON = ConvertTo-Json -InputObject ([object[]]$CommandArgs) -Compress
    & $python -m infrastructure.local.exec_python
    $commandExitCode = $LASTEXITCODE
}
finally {
    if ($null -eq $previousLaunchArguments) {
        Remove-Item Env:MUTACO_LAUNCH_ARGUMENTS_JSON -ErrorAction SilentlyContinue
    }
    else {
        $env:MUTACO_LAUNCH_ARGUMENTS_JSON = $previousLaunchArguments
    }
    $ErrorActionPreference = $previousErrorActionPreference
    Pop-Location
}
exit $commandExitCode
