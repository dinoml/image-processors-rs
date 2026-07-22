# FFmpeg Development Libraries

The `video` feature uses `video-rs`, which links FFmpeg through `ffmpeg-next` and `ffmpeg-sys-next`. It needs FFmpeg development libraries, headers, and package metadata. An `ffmpeg.exe` on `PATH` is not enough.

## Package Manager Builds

Runtime packages such as FFmpeg essentials/full builds are useful for running `ffmpeg`, `ffprobe`, and shared DLLs. They are not sufficient for this crate unless the installed package also includes:

- C headers under an `include` directory
- import libraries under a `lib` directory
- either `*.pc` files under `lib/pkgconfig` or a layout usable through `FFMPEG_DIR`

`ffmpeg-sys-next` checks `FFMPEG_DIR`, then vcpkg, then pkg-config. If a choco/scoop/winget package provides a complete development/shared layout, it can work by setting:

```powershell
$env:FFMPEG_DIR = "C:\path\to\ffmpeg"
cargo check --workspace --features video
```

If the package only installs `bin\ffmpeg.exe` and DLLs, it is a runtime install, not a Rust linkable development install.

## Windows Setup

Install Git if it is missing:

```powershell
winget install --id Git.Git -e
```

Install FFmpeg development libraries through vcpkg:

```powershell
.\scripts\setup-vcpkg-ffmpeg.ps1 -CheckCargo
```

For later shells, set:

```powershell
$env:VCPKG_ROOT = (Resolve-Path .\.vcpkg)
$env:VCPKG_DEFAULT_TRIPLET = "x64-windows"
$env:PKG_CONFIG_PATH = (Resolve-Path .\vcpkg_installed\x64-windows\lib\pkgconfig)
$env:PATH = "$(Resolve-Path .\vcpkg_installed\x64-windows\bin);$env:PATH"
cargo check --workspace --features video
```

The repo includes `vcpkg.json`, so vcpkg installs the native FFmpeg dependency set from the repository root. The setup script exports `PKG_CONFIG_PATH` because vcpkg manifest mode places the installed files in `vcpkg_installed`.

## Linux Setup

Install FFmpeg development libraries and `pkg-config`. On Ubuntu:

```bash
sudo apt-get install clang libavcodec-dev libavdevice-dev libavfilter-dev \
  libavformat-dev libavutil-dev libclang-dev libswresample-dev \
  libswscale-dev pkg-config
```

## macOS Setup

Install FFmpeg and `pkg-config` with Homebrew:

```bash
brew install ffmpeg pkg-config
```

## CI

The video CI lanes install FFmpeg development libraries on Windows, Linux, and macOS, and run:

```text
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
```

The Windows lane also installs `pkgconfiglite`, because `ffmpeg-sys-next` uses
the pkg-config metadata from the repository's manifest-mode vcpkg installation.

Default and `url` feature checks do not require FFmpeg development libraries.
