# CGC llama.cpp — Windows (x86_64)

**Target:** Windows 10/11, x86_64, CPU-only (Intel/AMD)  
**Backend:** MinGW64 (MSYS2)  
**Expected (Qwen3.6 IQ3_XXS, -ngl 4):** 2-4 t/s decode

---

## Prerequisites

1. **MSYS2** installed at `C:\msys64`
2. **MinGW64 toolchain:**

```bash
pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-cmake mingw-w64-x86_64-ninja
```

## Quick Start

### Build from Source

```batch
cd deploy-harmonyos\windows
build-windows.bat
```

### Run

```bash
# Basic decode
./run.sh -m /path/to/model.gguf -ngl 4

# MTP speculative decode
./run.sh -m /path/to/model.gguf --mtp 2 -ngl 4

# OpenAI-compatible server
./run.sh -m /path/to/model.gguf --server

# With expert-cache (4GB)
./run.sh -m /path/to/model.gguf --expert-cache 4294967296 -ngl 4
```

### Benchmark

```bash
./benchmark.sh /path/to/model.gguf 4
```

### Package Release

```batch
package-windows.bat
```

Produces `llama-cpp-cgc-windows-x64.zip` with binaries + DLLs.

---

## Architecture

```
deploy-harmonyos/windows/
├── build-windows.bat         # Build from source (MSYS2 MinGW64)
├── run-windows.sh            # Run with auto DLL detection
├── benchmark-windows.sh      # A/B benchmark (basic vs expert-cache)
├── package-windows.bat       # Package release zip
└── README.md                 # This file
```

## Required DLLs (bundled or from MSYS2)

| DLL | Source | Purpose |
|-----|--------|---------|
| `libstdc++-6.dll` | MSYS2 MinGW64 | C++ runtime |
| `libwinpthread-1.dll` | MSYS2 MinGW64 | Threading |
| `libgomp-1.dll` | MSYS2 MinGW64 | OpenMP (parallelism) |
| `libgcc_s_seh-1.dll` | MSYS2 MinGW64 | GCC runtime |

If DLLs are not in the script directory, `run.sh` auto-falls back to `C:\msys64\mingw64\bin`.

## Performance Tuning

| Parameter | Value | Notes |
|-----------|-------|-------|
| `-ngl` | 4 | 2GB VRAM (MX250) or CPU offload |
| `-c` | 2048 | Context size |
| `-t` | 4 | CPU threads |
| `--expert-cache` | 2-4 GB | L4 skip-load for MoE expert caching |

## Known Limitations

- **No GPU offload on Intel UHD** — UMA=1, shared memory, no int dot support
- **MX250 limited to -ngl 4** — 2GB VRAM, only 4 layers fit
- **Dual GPU (-sm layer) is slower** — Intel UHD drags down MX250
- **Expert-cache helps** — Reduces memory pressure for MoE models
