"""
Exports a trained QLoRA checkpoint to GGUF format for llama.cpp inference.

Pipeline:
  1. Load the base model + LoRA adapter, merge them (Unsloth's merged_16bit
     save), producing a full HF-format model.
  2. Run llama.cpp's convert_hf_to_gguf.py on the merged model to produce
     an F16 GGUF file.
  3. Quantize the F16 GGUF down to a smaller format (default Q4_K_M) using
     llama.cpp's quantize binary, for faster/smaller inference.

Requires the llama.cpp clone from Week 1 (cloned outside the repo, e.g.
D:\Projects\llama.cpp) with both build-cuda and build-vulkan already built.
"""

import argparse
import os
import subprocess
import sys

_cwd = sys.path[0]
if _cwd in sys.path:
    sys.path.remove(_cwd)  # avoid local datasets/ folder shadowing pip's datasets
from unsloth import FastLanguageModel
sys.path.insert(0, _cwd)


def merge_adapter(checkpoint_dir: str, merged_output_dir: str, max_seq_length: int = 512):
    """Loads a LoRA checkpoint and saves a merged 16-bit HF model."""
    print(f"Loading checkpoint from {checkpoint_dir}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=checkpoint_dir,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        device_map={"": 0},
    )
    print(f"Merging adapter and saving to {merged_output_dir}...")
    model.save_pretrained_merged(merged_output_dir, tokenizer, save_method="merged_16bit")
    print("Merge complete.")


def convert_to_gguf(llama_cpp_dir: str, merged_model_dir: str, gguf_output_path: str):
    """Runs llama.cpp's convert_hf_to_gguf.py to produce an F16 GGUF file."""
    convert_script = os.path.join(llama_cpp_dir, "convert_hf_to_gguf.py")
    if not os.path.exists(convert_script):
        raise FileNotFoundError(f"convert_hf_to_gguf.py not found at {convert_script}")

    print(f"Converting {merged_model_dir} -> {gguf_output_path} (F16)...")
    subprocess.run([
        sys.executable, convert_script,
        merged_model_dir,
        "--outfile", gguf_output_path,
        "--outtype", "f16",
    ], check=True)
    print("Conversion to F16 GGUF complete.")


def quantize_gguf(llama_cpp_dir: str, f16_gguf_path: str, quantized_output_path: str,
                   quant_type: str = "Q4_K_M", backend: str = "cuda"):
    """
    Runs llama.cpp's quantize binary to shrink the F16 GGUF down to the
    requested quant type. backend selects which build folder's binary to use
    (build-cuda or build-vulkan from Week 1); the quantize tool itself is
    CPU-only regardless of backend, but binaries live in each build folder.
    """
    build_dir = "build-cuda" if backend == "cuda" else "build-vulkan"
    quantize_bin = os.path.join(llama_cpp_dir, build_dir, "bin", "Release", "llama-quantize.exe")
    if not os.path.exists(quantize_bin):
        raise FileNotFoundError(f"llama-quantize.exe not found at {quantize_bin}")

    print(f"Quantizing {f16_gguf_path} -> {quantized_output_path} ({quant_type})...")
    subprocess.run([quantize_bin, f16_gguf_path, quantized_output_path, quant_type], check=True)
    print("Quantization complete.")


def main():
    parser = argparse.ArgumentParser(description="Export a QLoRA checkpoint to quantized GGUF.")
    parser.add_argument("--checkpoint", required=True, help="Path to the trained LoRA checkpoint dir.")
    parser.add_argument("--llama-cpp-dir", default=r"D:\Projects\llama.cpp",
                         help="Path to the llama.cpp clone (with build-cuda/build-vulkan already built).")
    parser.add_argument("--output-dir", default="models/gguf_export",
                         help="Where to write merged model, F16 GGUF, and quantized GGUF.")
    parser.add_argument("--quant-type", default="Q4_K_M", help="Quantization type (e.g. Q4_K_M, Q5_K_M, Q8_0).")
    parser.add_argument("--backend", default="cuda", choices=["cuda", "vulkan"],
                         help="Which llama.cpp build to use for the quantize binary.")
    parser.add_argument("--skip-merge", action="store_true",
                         help="Skip the merge step (use if a merged model already exists at --output-dir/merged).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    merged_dir = os.path.join(args.output_dir, "merged")
    f16_path = os.path.join(args.output_dir, "model-f16.gguf")
    quantized_path = os.path.join(args.output_dir, f"model-{args.quant_type}.gguf")

    if not args.skip_merge:
        merge_adapter(args.checkpoint, merged_dir)

    convert_to_gguf(args.llama_cpp_dir, merged_dir, f16_path)
    quantize_gguf(args.llama_cpp_dir, f16_path, quantized_path, args.quant_type, args.backend)

    print(f"\nDone. Quantized GGUF at: {quantized_path}")


if __name__ == "__main__":
    main()