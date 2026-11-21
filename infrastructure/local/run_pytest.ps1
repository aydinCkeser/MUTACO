[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PytestArgs
)

$ErrorActionPreference = 'Stop'
$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$localAppData = [Environment]::GetFolderPath('LocalApplicationData')
if (-not $localAppData) {
    throw 'LOCALAPPDATA is unavailable; refusing to place pytest output in the repository'
}
$tempRoot = [IO.Path]::GetFullPath((Join-Path $localAppData 'MUTACO\pytest'))
if ($tempRoot.StartsWith($repositoryRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw "pytest temporary root resolves inside the repository: $tempRoot"
}
New-Item -ItemType Directory -Path $tempRoot -Force | Out-Null
$env:PYTEST_DEBUG_TEMPROOT = $tempRoot

$pythonArgs = @('-m', 'pytest') + $PytestArgs
& (Join-Path $PSScriptRoot 'run_mutaco.ps1') -Environment 'mutaco-sbi' -CommandArgs $pythonArgs
exit $LASTEXITCODE
