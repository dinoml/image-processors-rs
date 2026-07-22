param(
    [string]$VcpkgRoot = (Join-Path $PSScriptRoot "..\.vcpkg"),
    [string]$Triplet = "x64-windows",
    [switch]$CheckCargo
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "git is required to clone vcpkg. Install it with winget install --id Git.Git -e"
}

if (-not (Test-Path $VcpkgRoot)) {
    git clone https://github.com/microsoft/vcpkg $VcpkgRoot
}

$VcpkgRoot = Resolve-Path $VcpkgRoot
$Bootstrap = Join-Path $VcpkgRoot "bootstrap-vcpkg.bat"
$Vcpkg = Join-Path $VcpkgRoot "vcpkg.exe"

& $Bootstrap -disableMetrics
& $Vcpkg install --triplet $Triplet --x-manifest-root=$RepoRoot

$env:VCPKG_ROOT = $VcpkgRoot
$env:VCPKG_DEFAULT_TRIPLET = $Triplet
$PkgConfigPath = Join-Path $RepoRoot "vcpkg_installed\$Triplet\lib\pkgconfig"
$BinPath = Join-Path $RepoRoot "vcpkg_installed\$Triplet\bin"

if (Test-Path $PkgConfigPath) {
    if ($env:PKG_CONFIG_PATH) {
        $env:PKG_CONFIG_PATH = "$PkgConfigPath;$env:PKG_CONFIG_PATH"
    } else {
        $env:PKG_CONFIG_PATH = $PkgConfigPath
    }
}

if (Test-Path $BinPath) {
    $env:PATH = "$BinPath;$env:PATH"
}

Write-Host "VCPKG_ROOT=$env:VCPKG_ROOT"
Write-Host "VCPKG_DEFAULT_TRIPLET=$env:VCPKG_DEFAULT_TRIPLET"
Write-Host "PKG_CONFIG_PATH=$env:PKG_CONFIG_PATH"

if ($CheckCargo) {
    cargo check --workspace --features video
}
