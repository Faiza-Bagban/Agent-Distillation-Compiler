# GGUF Vulkan Backend Test -- Week 6 Thu

## Setup
Same quantized checkpoint as Wednesday's CUDA test
(`models/gguf_export/model-Q4_K_M.gguf`), tested via llama.cpp's Vulkan
build (`build-vulkan/bin/Release/llama-cli.exe`).

## Devices available
Vulkan0: AMD Radeon 740M Graphics (iGPU, 8055 MiB, 7652 MiB "free")
Vulkan1: NVIDIA GeForce RTX 4050 Laptop GPU (discrete, 5920 MiB, 5152 MiB free)

## Results

**Full offload (-ngl 99), device unspecified (defaulted to Vulkan0/iGPU):**
- Prompt: 13.8 t/s, Generation: 26.0 t/s
- Output coherent and correct.

**Full offload (-ngl 99), explicit --device Vulkan0 (iGPU):**
- Failed: `ggml_vulkan: Device memory allocation of size 1024 failed`
  (`ErrorOutOfDeviceMemory`). The iGPU reports shared system RAM as "free"
  but hit a practical single-allocation limit trying to fit all 32 layers.

**Partial offload (-ngl 20), explicit --device Vulkan0 (iGPU):**
- Prompt: 7.3 t/s, Generation: 5.2 t/s
- Output coherent and correct, noticeably slower than full-GPU runs (expected
  for a weaker integrated GPU with less effective VRAM headroom).

## Interpretation
- The iGPU can run this model, but only with partial layer offload -- full
  offload exceeds what it can practically allocate despite showing enough
  "free" memory on paper.
- Vulkan on the discrete RTX 4050 (Wednesday's default, unconfirmed device
  in that run) performed close to CUDA speeds; a same-device Vulkan vs CUDA
  comparison would need an explicit --device Vulkan1 run to be sure.

## Next step (Fri)
Log a clean side-by-side table: CUDA vs Vulkan on RTX 4050, and Vulkan on
the iGPU, all with matched settings, for the final cross-backend comparison.